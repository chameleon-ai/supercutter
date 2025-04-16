import argparse
import datetime
from enum import Enum
import json
import mimetypes
import os
import re
import subprocess
import traceback
from math import gcd
from functools import reduce

class MatchMode(Enum):
    segment = 'segment' # Get the timestamp range of the full segment
    word = 'word' # Get the timestamps of only the matched pattern or word
    def __str__(self):
        return self.value

# Format a timedelta into hh:mm:ss.ms
def format_timedelta(ts : datetime.timedelta):
    hours, rm_hr = divmod(ts.total_seconds(), 3600)
    mins, rm_min = divmod(rm_hr, 60)
    sec, rm_sec = divmod(rm_min, 1)
    # Omit leading zero hours and minutes
    if int(hours) == 0:
        if int(mins) == 0:
            ts_str = '{:02}.{:03}'.format(int(sec), int(rm_sec*1000))
        else:
            ts_str = '{:02}:{:02}.{:03}'.format(int(mins), int(sec), int(rm_sec*1000))
    else:
        ts_str = '{:02}:{:02}:{:02}.{:03}'.format(int(hours), int(mins), int(sec), int(rm_sec*1000))
    return ts_str

def build_filter_graph(segment_list : list, scale = None, setdar = None):
    # Reference: https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
    vtrim_graph = ''
    atrim_graph = ''
    total_segments = []
    for video_index, segments in enumerate(segment_list):
        # Build a filter graph on the kept segments
        # Nice reference for how to build a filter graph: https://github.com/sriramcu/ffmpeg_video_editing
        for index, segment in enumerate(segments, start=1):
            segment_start, segment_end = segment
            total_segments.append(segment)
            # [0]trim=start=34.5:end=55.1,setpts=PTS-STARTPTS[v1];
            vtrim_graph += f'[{video_index}]'
            if scale is not None:
                vtrim_graph += f'scale={scale},'
            if setdar is not None:
                vtrim_graph += f'setdar={setdar},'
            vtrim_graph += f'trim=start={segment_start}:end={segment_end},setpts=PTS-STARTPTS[v{len(total_segments)}];'
            atrim_graph += f'[{video_index}]atrim=start={segment_start}:end={segment_end},asetpts=PTS-STARTPTS[a{len(total_segments)}];'
            #print('{} {}-{}'.format(index, segment_start, segment_end))
    vconcat_graph = ''
    for index, _ in enumerate(total_segments, start=1):
        # [v1][v2][v3]concat=n=3:v=1:a=0[outv]
        vconcat_graph += f'[v{index}]'
    vconcat_graph += f'concat=n={len(total_segments)}:v=1:a=0[outv]'
    aconcat_graph = ''
    for index, _ in enumerate(total_segments, start=1):
        aconcat_graph += f'[a{index}]'
    aconcat_graph += f'concat=n={len(total_segments)}:v=0:a=1[outa]'

    filter_graph = vtrim_graph + vconcat_graph + ';' + atrim_graph + aconcat_graph
    #print(filter_graph)
    return filter_graph

def parse_aspect_ratio(ratio):
    """Convert aspect ratio string to a tuple of integers (numerator, denominator)."""
    numerator, denominator = map(int, ratio.split(':'))
    return numerator, denominator

def find_common_scale_and_dar(input_filenames : list):
    common_dar = None
    common_scale = None
    dars = []
    resolutions = []
    widths = []
    for video in input_filenames:
        result = subprocess.run(["ffprobe","-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,sample_aspect_ratio,display_aspect_ratio", "-of", "csv=p=0", video], stdout=subprocess.PIPE, text=True)
        width, height, sar, dar = [x for x in result.stdout.strip().split(',')]
        dars.append(dar)
        widths.append(width)
        resolutions.append(f'{width}x{height}')
        #print(f'{width} x {height} {sar} {dar}')
    if len(set(dars)) > 1: # Test to see if all dars are the same by using a set, if the set length is more than 1, there is a difference
        ratios = [parse_aspect_ratio(ratio) for ratio in dars]
        # Calculate the GCD of all numerators and denominators
        common_numerator = reduce(lambda x, y: gcd(x, y), [num for num, _ in ratios])
        common_denominator = reduce(lambda x, y: gcd(x, y), [den for _, den in ratios])
        # Simplify the common ratio
        common_gcd = gcd(common_numerator, common_denominator)
        common_dar = f'{common_numerator // common_gcd}/{common_denominator // common_gcd}'
        print(f'Common display aspect ratio:{common_dar}')
    if len(set(resolutions)) > 1: # Same for resolution
        common_scale = f'{min(widths)}:-1'
        print(f'Common resolution: {common_scale}')
    return common_scale, common_dar

def segment_video(input_filenames : list, filter_graph : str):
    ffmpeg_args = ['ffmpeg', '-hide_banner', '-y']
    for input_filename in input_filenames:
        # Input file(s) to process
        ffmpeg_args.extend(['-i', input_filename])
    # Build the filter arguments
    ffmpeg_args.extend(['-filter_complex', filter_graph, '-map', '[outv]', '-map', '[outa]'])

    # Encoder. This is used to generate a temporary file.
    ffmpeg_args.extend(["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"])
    ffmpeg_args.extend(["-c:a", "libopus", "-b:a", "320k"])
    
    # Output file
    output_filename = 'out.mkv'
    ffmpeg_args.append(output_filename)
    print('Rendering cut video (please be patient)...')
    print(' '.join(ffmpeg_args))
    # Use popen so we can pend on completion
    pope = subprocess.Popen(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in iter(pope.stderr.readline, ""):
        if 'frame=' in line:
            print('\r' + line.strip(), end='')
        else:
            print(line, end='')
    pope.stderr.close()
    pope.wait()
    if pope.returncode != 0:
        raise RuntimeError('ffmpeg returned code {}'.format(pope.returncode))

    if os.path.isfile(output_filename):
        return output_filename
    else:
        raise RuntimeError("File '{}' not found".format(output_filename))

def get_video_duration(input_filename : str, start_time = 0.0):
    """
    Uses ffprobe to find out how long the input is.

    input_filename - The file to analyze
    start_time - This relative start time will be subtracted from the total duration if specified
    """
    # https://superuser.com/questions/650291/how-to-get-video-duration-in-seconds
    result = subprocess.run(["ffprobe","-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_filename], stdout=subprocess.PIPE, text=True)
    duration_seconds = float(result.stdout)
    return datetime.timedelta(seconds=duration_seconds - start_time)

def render_segments(ts_list : list, adjust_pre_ms = 100, adjust_post_ms = 300):
    video_segments = []
    
    for i, (input_filename, timestamps) in enumerate(ts_list):
        adjusted_timestamps = [ [t[0], t[1]] for t in timestamps]
        if adjust_pre_ms > 0 or adjust_post_ms > 0: # Add time to the timestamps
            pre_sec = adjust_pre_ms / 1000.0 # Amount of time to adjust the start timestamp back
            post_sec = adjust_post_ms / 1000.0 # Amount of time to adjust the end timestamp forward
            end = get_video_duration(input_filename).total_seconds()  # Cap at the duration limit
            for idx,(start_ts, end_ts) in enumerate(timestamps):
                prev_end = 0.0 if idx < 1 else timestamps[idx-1][1]
                # Clamp to prev segment end and next segment start
                adjusted_timestamps[idx][0] = prev_end if start_ts - pre_sec < prev_end else start_ts - pre_sec
                next_start = end if idx + 1 >= len(timestamps) else timestamps[idx+1][0]
                adjusted_timestamps[idx][1] = next_start if end_ts + post_sec > next_start else end_ts + post_sec
        
        # Make a new list with merged timestamps that overlap
        merged = [(adjusted_timestamps[0][0], adjusted_timestamps[0][1])] # Initialize the merged list with the first timestamp

        for current_start, current_end in adjusted_timestamps[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end: # Check if there is an overlap
                # Merge the intervals by updating the end time
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, just add the current timestamp
                merged.append((current_start, current_end))
        video_segments.append(merged)
    video_filenames = [video for video,_ in ts_list]
    scale, dar = find_common_scale_and_dar(video_filenames)
    filter_graph = build_filter_graph(video_segments, scale=scale, setdar=dar)
    input_filenames = [input_filename for input_filename,_ in ts_list] # Extract just the list of filenames
    return segment_video(input_filenames, filter_graph)

def search_transcript(transcript, pattern : str, match_mode = MatchMode.word):
    """
    Finds the match pattern in the transcript. Returns a list of datetime.timedeltas in the form of (start, end).
    """
    match = re.search(pattern, transcript['text'], re.IGNORECASE)
    timestamps = []
    if match is None:
        print('Search pattern not found in transcript.')
    else:
        for segment in transcript['segments']:
            if re.search(pattern, segment['text'], re.IGNORECASE):
                if match_mode == MatchMode.segment: # Retrieve timestamps of whole segment
                    timestamps.append((float(segment['start']), float(segment['end'])))
                elif match_mode == MatchMode.word:
                    for word in segment['words']: # Retrieve timestamps of each matching word
                        if re.search(pattern, word['text'], re.IGNORECASE):
                            timestamps.append((float(word['start']), float(word['end'])))
                            found = True
                    if not found: # Fallback to full segment if nothing was found at the word level
                        timestamps.append((float(segment['start']), float(segment['end'])))
    return timestamps

# Return a list of filenames along with the transcripts
def load_transcripts(input_filenames):
    file_transcript_pairs = []
    # Search for existing transcripts
    for video in input_filenames:
        transcript_filename = os.path.splitext(video)[0] + '.json'
        if os.path.isfile(transcript_filename):
            with open(transcript_filename) as fin:
                transcript = json.load(fin)
                file_transcript_pairs.append((video, transcript))
                print(f'Transcript "{transcript_filename}" was loaded from file.')
        else: # No transcript found
            file_transcript_pairs.append((video, None))
    return file_transcript_pairs

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            prog='Supercutter',
            description='Makes supercuts.')
        parser.add_argument('--condition_on_previous', action='store_true', help='Whether to provide the previous output as a prompt for the next window.')
        parser.add_argument('--detect_disfluencies', action='store_true', help='Enables disfluency detection.')
        parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Cuda device to use.')
        parser.add_argument('--find', type=str, help='String to search for in the transcript.')
        parser.add_argument('--language', type=str, default='en', choices=['auto', 'en', 'ja', 'fr'], help='Transcription language. Usually faster if you specify. Capability depends on the model.')
        parser.add_argument('--load_transcript', type=str, help='Skip whisper and operate on a pre-saved transcript file.')
        parser.add_argument('--match_mode', type=MatchMode, default='word', choices=list(MatchMode), help='Whether to match timestamps for the whole segment or just the word.')
        parser.add_argument('--model', type=str, default='large-v3-turbo', help='Whisper model to use.')
        parser.add_argument('--no_vad', action='store_false', help='Do not use Voice Activity Detection')
        parser.add_argument('--no_naive', action='store_false', help='Do not use naive approach')
        parser.add_argument('--prompt', type=str, help='Initial prompt to use for transcription.')
        args, unknown_args = parser.parse_known_args()
        if help in args:
            parser.print_help()

        # Flip some negative flags for clarity
        vad = not args.no_vad
        naive_approach = not args.no_naive

        # Traverse all arguments for files and directories
        input_files = []
        for arg in unknown_args:
            if os.path.isfile(arg):
                input_files.append(arg)
            elif os.path.isdir(arg):
                input_files.extend([os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(arg) for f in filenames])

        # Take only videos from all listed files and sort
        try:
            import natsort
            input_videos = natsort.natsorted([file for file in input_files if mimetypes.guess_type(file)[0].split('/')[0] == 'video'])
        except ImportError: # Fallback to standard library sort
            print('natsort not found. Falling back to standard sort. Try "pip install natsort" if you want files to be naturally sorted.')
            input_videos = sorted([file for file in input_files if mimetypes.guess_type(file)[0].split('/')[0] == 'video'])
        
        # Load transcripts if possible
        video_transcript_pairs = load_transcripts(input_videos)
        missing_transcript = any(transcript is None for _, transcript in video_transcript_pairs)
        if missing_transcript: # Only transcribe if there are missing transcripts
            loaded_transcripts = []
            print('Loading whisper...')
            import whisper_timestamped as whisper
            whisper_model = whisper.load_model(args.model, device=args.device, download_root="./models/")
            for video, transcript in video_transcript_pairs:
                if transcript is None:
                    print(f'Transcribing file "{video}"')
                    audio = whisper.load_audio(video)
                    result = whisper.transcribe_timestamped(whisper_model,
                        audio,
                        naive_approach=naive_approach,
                        initial_prompt=args.prompt,
                        vad='auditok' if vad else False,
                        beam_size=5,
                        best_of=5,
                        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        language=None if args.language == 'auto' else args.language,
                        condition_on_previous_text=args.condition_on_previous,
                        detect_disfluencies=args.detect_disfluencies)
                    loaded_transcripts.append((video,result))
                    transcript_filename = os.path.splitext(video)[0] + '.json'
                    print(f'Saving transcript to file {transcript_filename}')
                    with open(transcript_filename, 'w', encoding='utf-8') as fout:
                        json.dump(result, fout, ensure_ascii=False, indent=2)
                else:
                    loaded_transcripts.append((video, transcript))
        else:
            loaded_transcripts = video_transcript_pairs
        
        # Search the transcripts for the specified pattern
        if args.find:
            timestamps = []
            for video, transcript in loaded_transcripts:
                print(f'Searching transcript of "{video}"')
                ts = search_transcript(transcript, args.find, args.match_mode)
                if len(ts) > 0: # Only append the file if matches were found
                    print('Count: {}'.format(len(ts)))
                    timestamps.append((video, ts))
            if len(timestamps) > 0:
                output_filename = render_segments(timestamps)
                print(f'Rendered file: {output_filename}')
    except argparse.ArgumentError as e:
        print(e)
    except Exception:
        print(traceback.format_exc())