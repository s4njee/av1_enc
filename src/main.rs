use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use walkdir::WalkDir;
use std::error::Error;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use clap::Parser;
use rayon::ThreadPoolBuilder;
use num_cpus;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use regex::Regex;
use std::sync::Mutex;
use std::collections::HashSet;
use crossbeam_channel::{bounded, Receiver};
use std::thread;
use std::io::Read;  // Add this to your imports

enum ControlMessage {
    UpdateThreads(usize),
    Exit,
}

fn read_char() -> std::io::Result<char> {
    let mut buffer = [0; 1];
    std::io::stdin().read_exact(&mut buffer)?;
    Ok(buffer[0] as char)
}

fn spawn_control_thread() -> Receiver<ControlMessage> {
    let (tx, rx) = bounded::<ControlMessage>(10);
    
    // Spawn thread to handle user input
    thread::spawn(move || {
        println!("\nDynamic controls:");
        println!("'+' to increase parallel encodes");
        println!("'-' to decrease parallel encodes");
        println!("'q' to quit\n");

        loop {
            if let Ok(input) = read_char() {
                match input {
                    '+' => {
                        let current = rayon::current_num_threads();
                        if let Ok(_) = rayon::ThreadPoolBuilder::new()
                            .num_threads(current + 1)
                            .build_global() {
                            println!("\nIncreased to {} parallel encodes", current + 1);
                            let _ = tx.send(ControlMessage::UpdateThreads(current + 1));                        }
                    },
                    '-' => {
                        let current = rayon::current_num_threads();
                        if current > 1 {
                            if let Ok(_) = rayon::ThreadPoolBuilder::new()
                                .num_threads(current - 1)
                                .build_global() {
                                println!("\nDecreased to {} parallel encodes", current - 1);
                                let _ = tx.send(ControlMessage::UpdateThreads(current - 1));                            }
                        }
                    },
                    'q' => break,
                    _ => {}
                }
            }
        }
    });

    rx
}

// Add a new struct to handle logging
struct CompletedFiles {
    log_path: PathBuf,
    completed: HashSet<String>,
    file_handle: Mutex<fs::File>,
}

impl CompletedFiles {
    fn new(base_dir: &str) -> Result<Self, Box<dyn Error>> {
        let log_path = Path::new(base_dir).join("completed_encodes.log");
        let mut completed = HashSet::new();
        
        // Read existing log if it exists
        if log_path.exists() {
            let contents = fs::read_to_string(&log_path)?;
            completed = contents.lines().map(String::from).collect();
        }
        
        // Open file in append mode
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
            
        Ok(CompletedFiles {
            log_path,
            completed,
            file_handle: Mutex::new(file),
        })
    }
    
    fn is_completed(&self, path: &str) -> bool {
        self.completed.contains(path)
    }
    
    fn mark_completed(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut file = self.file_handle.lock().unwrap();
        writeln!(file, "{}", path)?;
        file.flush()?;
        Ok(())
    }
}

#[derive(Parser)]
#[command(name = "av1_enc")]
#[command(about = "Converts MP4 files to AV1 format using SVT-AV1")]
struct Args {
    /// Base directory to scan for video files
    #[arg(default_value = ".")]
    base_dir: String,

    #[arg(short = 'e', long, value_delimiter = ',', default_value = "mp4")]
    extensions: Vec<String>,

    /// Output directory (will be created if it doesn't exist)
    #[arg(short = 'o', long)]
    output_dir: Option<String>,

    /// Custom ffmpeg command (will substitute -i input and output.mkv automatically)
    #[arg(short = 'f', long)]
    ffmpeg_command: Option<String>,

    /// SVT-AV1 preset (0-13, lower is better quality but slower)
    #[arg(short, long, default_value = "8")]
    preset: u32,

    /// Constant Rate Factor (0-63, lower is better quality)
    #[arg(short = 'c', long, default_value = "30")]
    crf: u32,

    /// Tune parameter (default: 0)
    /// 0=vq, 1=psnr, 2=ssim
    #[arg(short = 't', long, default_value = "0")]
    tune: u32,

    /// Film grain synthesis level (0-50, default: 0)
    #[arg(short = 'g', long, default_value = "0")]
    film_grain: u32,

    /// Video filter string (e.g., "scale=1920:1080")
    #[arg(short = 'v', long)]
    vf: Option<String>,

   /// Audio bitrate in kbps (e.g., 128)
    #[arg(short = 'b', long)]
    audio_bitrate: Option<u32>,

    /// Copy audio stream instead of re-encoding
    #[arg(short = 'k', long)]
    copy_audio: bool,

    /// Number of parallel videos to process (default: number of CPU cores)
    #[arg(short = 'j', long, default_value_t = num_cpus::get())]
    threads: usize,

    /// Delete original files after successful conversion
    #[arg(short = 'd', long)]
    delete_originals: bool,

    #[arg(short = 'y', long)]
    force_yes: bool,

    #[arg(long)]
    no_log: bool,

    /// Enable anime-optimized encoding settings
    #[arg(long)]
    anime: bool,
}

struct VideoInfo {
    path: String,
}

// Add helper function:
fn check_overwrites(
    video_infos: &[VideoInfo], 
    output_dir: Option<&str>, 
    delete_originals: bool
) -> Vec<PathBuf> {
    if delete_originals {
        return Vec::new(); // No overwrite checks needed if we're deleting originals
    }

    video_infos.iter()
        .filter_map(|video| {
            let input_path = Path::new(&video.path);
            let input_stem = input_path.file_stem().unwrap();
            
            let final_filename = if input_path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("mkv")) {
                input_path.file_name().unwrap().to_string_lossy().to_string()
            } else {
                format!("{}.mkv", input_stem.to_string_lossy())
            };
            
            let final_path = match output_dir {
                Some(dir) => Path::new(dir).join(&final_filename),
                None => input_path.with_file_name(&final_filename),
            };

            if final_path.exists() {
                Some(final_path)
            } else {
                None
            }
        })
        .collect()
}


fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_arguments(&args)?;

    // Initialize logging
    let completed_files = if !args.no_log {
        Some(Arc::new(CompletedFiles::new(&args.base_dir)?))
    } else {
        None
    };

    // Configure thread pool
    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    // Setup control thread
    let control_rx = spawn_control_thread();

    // Check if base directory exists
    if !Path::new(&args.base_dir).exists() {
        eprintln!("Error: Directory '{}' does not exist", args.base_dir);
        std::process::exit(1);
    }

    // Create output directory if specified and doesn't exist
    if let Some(ref output_dir) = args.output_dir {
        fs::create_dir_all(output_dir)?;
        println!("Output directory: {}", output_dir);
    }

    println!("Starting AV1 encoding...");
    println!("Base directory: {}", args.base_dir);
    println!("File extensions: {}", args.extensions.join(", "));
    println!("SVT-AV1 preset: {}", args.preset);
    println!("CRF: {}", args.crf);
    println!("Tune: {} ({})", args.tune, match args.tune {
        0 => "vq",
        1 => "psnr",
        2 => "ssim",
        _ => "unknown"
    });
    if let Some(ref vf) = args.vf {
        println!("Video filter: {}", vf);
    }
    if args.copy_audio {
        println!("Audio: Copy original streams");
    } else {
        println!("Audio: Opus encode{}", args.audio_bitrate.map_or(String::new(), |br| format!(" @ {}k", br)));
    }
    println!("Using {} parallel video processes", args.threads);
    if args.delete_originals {
        println!("Original files will be deleted after conversion");
    }
    let start_time = std::time::Instant::now();
    println!("Scanning for MKV/MP4 files and counting frames...");
    let (video_infos, total_frames, total_files) = scan_video_files(&args.base_dir, &args.extensions, completed_files.as_ref())?;
    // Get count BEFORE spawning thread
    let total_videos = video_infos.len();

    if video_infos.is_empty() {
        println!("No matching files found in directory: {}", args.base_dir);
        return Ok(());
    }

    println!("Found {} total files ({} successfully scanned) with {} total frames to encode", 
             total_files, video_infos.len(), total_frames);


    // Check for potential overwrites before starting
    let overwrites = check_overwrites(&video_infos, args.output_dir.as_deref(), args.delete_originals);
    if !overwrites.is_empty() && !args.force_yes {
        println!("\nThe following files would be overwritten:");
        for path in &overwrites {
            println!("  {}", path.display());
        }
        println!();
        
        if !prompt_yes_no("Do you want to continue and overwrite these files?") {
            println!("Aborting.");
            std::process::exit(0);
        }
    }

    println!("\nStarting AV1 encoding...");

    // Setup progress bar
    let pb = Arc::new(ProgressBar::new(total_frames as u64));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames ({percent}%) {msg}")
        .unwrap()
        .progress_chars("#>-"));


    // Counters
    let successful_conversions = Arc::new(AtomicUsize::new(0));
    let successful_frames = Arc::new(AtomicUsize::new(0));
    let deleted_files = Arc::new(AtomicUsize::new(0));
    let should_delete = args.delete_originals;

        // Clone them before spawning thread
    let successful_frames_thread = successful_frames.clone();
    let deleted_files_thread = deleted_files.clone();
    let successful_conversions_thread = successful_conversions.clone();
    let pb_thread = pb.clone();


    // Process files in parallel

      // Channel to signal when a video is complete
      let (video_tx, video_rx) = bounded::<()>(1);

      // Spawn video processing in separate thread
    video_infos.par_iter()
        .for_each(|video| {


        // Skip if already completed
        if let Some(ref completed) = completed_files {
            if completed.is_completed(&video.path) {
                pb_thread.println(format!("Skipping already encoded file: {}", video.path));
                return;
            }
        }

        // else encode
         match convert_to_av1(
	    &video.path, 
	    args.output_dir.as_deref(),
	    args.ffmpeg_command.as_deref(),
	    args.preset,
	    args.crf,
	    args.tune,
	    args.film_grain,
        args.anime,
	    args.vf.as_deref(),
	    args.copy_audio,
	    args.audio_bitrate,
        completed_files.clone(),
	    &pb_thread,
	    successful_frames_thread.clone(),
        should_delete,
        deleted_files_thread.clone()
	){ 
        Ok(_) => {
            successful_conversions_thread.fetch_add(1, Ordering::Relaxed);
        },
        Err(e) => {
            eprintln!("Failed to convert {}: {}", video.path, e);
        }
    } 
    let _ = video_tx.send(());  // Signal completion
});
    let mut remaining_videos = total_videos;  // Use our saved count
    // Monitor both video completion and control messages
    loop {
        crossbeam_channel::select! {
            recv(control_rx) -> msg => match msg {
                Ok(ControlMessage::UpdateThreads(count)) => {
                    if let Ok(_) = ThreadPoolBuilder::new()
                        .num_threads(count)
                        .build_global() {
                        println!("\nUpdated to {} parallel encodes", count);
                    }
                },
                Ok(ControlMessage::Exit) => break,
                Err(_) => break,
            },
            recv(video_rx) -> _ => {
                remaining_videos -= 1;
                if remaining_videos == 0 {
                    break;
                }
            }
        }
    }
    
    pb.finish_with_message("Encoding complete!");

    let converted_files = successful_conversions.load(Ordering::Relaxed);
    let converted_frames = successful_frames.load(Ordering::Relaxed);
    let deleted_count = deleted_files.load(Ordering::Relaxed);
    let duration = start_time.elapsed();

    println!("\nEncoding Summary:");
    println!("Total files processed: {}", total_files);
    println!("Successfully converted: {}", converted_files);
    println!("Failed conversions: {}", total_files - converted_files);
    println!("Total frames encoded: {}/{}", converted_frames, total_frames);
    if args.delete_originals {
        println!("Original files deleted: {}", deleted_count);
    }
    println!("Time taken: {:.2} seconds", duration.as_secs_f64());

    Ok(())
}

fn convert_to_av1(
    input_path: &str,
    output_dir: Option<&str>,
    ffmpeg_command: Option<&str>,
    preset: u32,
    crf: u32,
    tune: u32,
    film_grain: u32,
    anime: bool,
    vf: Option<&str>,
    copy_audio: bool,
    audio_bitrate: Option<u32>,
    completed_files: Option<Arc<CompletedFiles>>,
    pb: &ProgressBar,
    frame_counter: Arc<AtomicUsize>,
    should_delete: bool,
    deleted_files: Arc<AtomicUsize>
) -> Result<(), Box<dyn Error>> {
    let input_path = Path::new(input_path);
    let input_stem = input_path.file_stem().unwrap().to_string_lossy();
    
    let temp_filename = format!("{}-temp-{}.mkv", input_stem, std::process::id());
    let temp_path = match output_dir {
        Some(dir) => Path::new(dir).join(&temp_filename),
        None => input_path.with_file_name(&temp_filename),
    };

    let mut cmd = Command::new("ffmpeg");

    if let Some(custom_cmd) = ffmpeg_command {
        // Handle custom command case...
        cmd.arg("-i")
            .arg(input_path)
            .args(custom_cmd.split_whitespace())
            .arg("-progress")
            .arg("pipe:1")
            .arg("-y")
            .arg(&temp_path);
    } else {
        // Get audio channels for first two tracks
        let audio_channels_1 = get_audio_channels(input_path)?;
        let audio_channels_2 = get_second_audio_channels(input_path).unwrap_or(2); // Default to stereo if no second track

        cmd.arg("-i")
            .arg(input_path)
            .arg("-map")
            .arg("0:v")   // Map video
            .arg("-map")  // Map first audio track
            .arg("0:a:0")
            .arg("-map") // Map second audio track
            .arg("0:a:1")
            .arg("-map")  // Map all subtitle tracks
            .arg("0:s?")
            .arg("-c:v")
            .arg("libsvtav1")
            .arg("-preset")
            .arg(preset.to_string())
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-svtav1-params")
            .arg(format!("tune={}:film-grain={}", tune, film_grain))
            .arg("-pix_fmt")
            .arg("yuv420p10le");

        // Add video filter if specified
        if let Some(filter) = vf {
            cmd.arg("-vf").arg(filter);
        }

        // Handle audio encoding for both tracks
        if copy_audio {
            cmd.arg("-c:a").arg("copy");
        } else {
            // First audio track
            configure_opus_track(&mut cmd, 0, audio_channels_1, audio_bitrate);
            // Second audio track
            configure_opus_track(&mut cmd, 1, audio_channels_2, audio_bitrate);
        }

        // Copy subtitles and other streams
        cmd.arg("-c:s").arg("copy")
           .arg("-c:t").arg("copy")
           .arg("-c:d").arg("copy");

        cmd.arg("-progress")
            .arg("pipe:1")
            .arg("-y")
            .arg(&temp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
    }

    // Rest of the function remains the same...
    let mut process = cmd.spawn()?;
    // ... (rest of existing function)
    
    Ok(())
}

fn configure_opus_track(cmd: &mut Command, track: u32, channels: u32, audio_bitrate: Option<u32>) {
    cmd.arg(format!("-c:a:{}", track)).arg("libopus");
    
    match channels {
        1 => {
            cmd.arg(format!("-ac:{}", track)).arg("1")
               .arg(format!("-b:a:{}", track)).arg(format!("{}k", audio_bitrate.unwrap_or(64)));
        },
        2 => {
            cmd.arg(format!("-ac:{}", track)).arg("2")
               .arg(format!("-b:a:{}", track)).arg(format!("{}k", audio_bitrate.unwrap_or(96)));
        },
        channels if channels >= 8 => {
            cmd.arg(format!("-ac:{}", track)).arg("8")
               .arg(format!("-b:a:{}", track)).arg(format!("{}k", audio_bitrate.unwrap_or(512)))
               .arg("-mapping_family").arg("1")
               .arg("-apply_phase_inv").arg("0")
               .arg("-channel_layout").arg("7.1");
        },
        channels if channels >= 6 => {
            cmd.arg(format!("-ac:{}", track)).arg("6")
               .arg(format!("-b:a:{}", track)).arg(format!("{}k", audio_bitrate.unwrap_or(384)))
               .arg("-mapping_family").arg("1")
               .arg("-apply_phase_inv").arg("0")
               .arg("-channel_layout").arg("5.1");
        },
        _ => {
            cmd.arg(format!("-ac:{}", track)).arg("2")
               .arg(format!("-b:a:{}", track)).arg(format!("{}k", audio_bitrate.unwrap_or(96)));
        },
    }
}

fn get_second_audio_channels(path: &Path) -> Result<u32, Box<dyn Error>> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("a:1")  // Select second audio stream
        .arg("-show_entries")
        .arg("stream=channels")
        .arg("-of")
        .arg("json")
        .arg(path)
        .output()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    
    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
        if let Some(stream) = streams.first() {
            if let Some(channels) = stream.get("channels").and_then(|c| c.as_u64()) {
                return Ok(channels as u32);
            }
        }
    }

    Err("No second audio track found".into())
}
fn get_frame_count(path: &Path) -> Result<usize, Box<dyn Error>> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-show_entries")
        .arg("stream=r_frame_rate")
        .arg("-of")
        .arg("json")
        .arg(path)
        .output()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    
    // Get duration from format section
    let duration = json
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .ok_or("Missing duration field")?
        .parse::<f64>()?;

    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
        if let Some(stream) = streams.first() {

            // Parse frame rate which comes as "num/den" string
            let r_frame_rate = stream.get("r_frame_rate")
                .and_then(|r| r.as_str())
                .ok_or("Could not get frame rate")?;
            
            let parts: Vec<f64> = r_frame_rate.split('/')
                .map(|p| p.parse::<f64>())
                .collect::<Result<Vec<f64>, _>>()?;
            
            if parts.len() == 2 && parts[1] != 0.0 {
                let fps = parts[0] / parts[1];
                let frame_count = (duration * fps).round() as usize;
                return Ok(frame_count);
            }
        }
    }

    // If we can't calculate from duration and frame rate, try container parameters
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-count_packets")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=nb_read_packets")
        .arg("-print_format")
        .arg("json")
        .arg("-i")
        .arg(path)
        .output()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    
    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
        if let Some(stream) = streams.first() {
            if let Some(packets) = stream.get("nb_read_packets")
                .and_then(|p| p.as_str())
                .and_then(|p| p.parse::<usize>().ok()) {
                if packets > 0 {
                    return Ok(packets);
                }
            }
        }
    }

    // If all else fails, estimate based on video duration and container frame rate
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-print_format")
        .arg("json")
        .arg("-i")
        .arg(path)
        .output()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    if let Some(duration) = json.get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|d| d.parse::<f64>().ok()) {
        // Assume common frame rate if we can't get it otherwise
        let estimated_frames = (duration * 24.0).round() as usize;
        if estimated_frames > 0 {
            return Ok(estimated_frames);
        }
    }

    Err("Could not determine frame count".into())
}

fn is_ffmpeg_installed() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn is_ffprobe_installed() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}


fn scan_video_files(
    base_dir: &str, 
    extensions: &[String],
    completed_files: Option<&Arc<CompletedFiles>>
) -> Result<(Vec<VideoInfo>, usize, usize), Box<dyn Error>> {
    let frame_scanning_pb = ProgressBar::new_spinner();
    frame_scanning_pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} Scanning: {msg} ({pos} files)")
        .unwrap());
    frame_scanning_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Convert all extensions to lowercase once
    let extensions: Vec<String> = extensions.iter()
        .map(|ext| ext.trim().to_lowercase())
        .collect();

    // First collect all matching files, excluding completed ones
    let matching_files: Vec<_> = WalkDir::new(base_dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            if let Some(ext) = entry.path().extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                let path_str = entry.path().to_string_lossy();
                
                // Check if file is in completed log
                if let Some(completed) = completed_files {
                    if completed.is_completed(&path_str) {
                        return false;
                    }
                }
                
                extensions.contains(&ext_lower)
            } else {
                false
            }
        })
        .collect();

    let total_files = matching_files.len();
    frame_scanning_pb.set_message("Analyzing files in parallel...");

    // Atomic counters for thread-safe counting
    let total_frames = Arc::new(AtomicUsize::new(0));
    let failed_files = Arc::new(AtomicUsize::new(0));
    let processed_files = Arc::new(AtomicUsize::new(0));

    // Process files in parallel
    let video_infos: Vec<_> = matching_files.par_iter()
        .filter_map(|entry| {
            let path_display = entry.path().display().to_string();
            frame_scanning_pb.set_message(path_display.clone());
            
            let result = match get_frame_count(entry.path()) {
                Ok(frame_count) => {
                    total_frames.fetch_add(frame_count, Ordering::Relaxed);
                    Some(VideoInfo {
                        path: entry.path().to_string_lossy().into_owned(),
                    })
                }
                Err(e) => {
                    eprintln!("Skipping {}: ffprobe failed: {}", path_display, e);
                    failed_files.fetch_add(1, Ordering::Relaxed);
                    None
                }
            };

            processed_files.fetch_add(1, Ordering::Relaxed);
            frame_scanning_pb.set_position(processed_files.load(Ordering::Relaxed) as u64);
            
            result
        })
        .collect();

    let final_total_frames = total_frames.load(Ordering::Relaxed);
    let final_failed_files = failed_files.load(Ordering::Relaxed);

    frame_scanning_pb.finish_with_message(format!(
        "Scanned {} files ({} skipped due to ffprobe failures)", 
        total_files,
        final_failed_files
    ));
    
    Ok((video_infos, final_total_frames, total_files))
}

fn prompt_yes_no(question: &str) -> bool {
    print!("{} [y/N] ", question);
    std::io::stdout().flush().unwrap();
    
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_ok() {
        input.trim().eq_ignore_ascii_case("y")
    } else {
        false
    }
}

fn get_audio_channels(path: &Path) -> Result<u32, Box<dyn Error>> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("a:0")  // Select first audio stream
        .arg("-show_entries")
        .arg("stream=channels")
        .arg("-of")
        .arg("json")
        .arg(path)
        .output()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    
    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
        if let Some(stream) = streams.first() {
            if let Some(channels) = stream.get("channels").and_then(|c| c.as_u64()) {
                return Ok(channels as u32);
            }
        }
    }

    // Default to stereo if we can't determine
    Ok(2)
}

fn validate_arguments(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.preset > 13 {
        return Err("Preset must be between 0 and 13".into());
    }
    if args.crf > 63 {
        return Err("CRF must be between 0 and 63".into());
    }
    if args.tune > 2 {
        return Err("Tune must be between 0 and 2".into());
    }
    if args.threads < 1 {
        return Err("Thread count must be at least 1".into());
    }
    if !is_ffmpeg_installed() {
        return Err("ffmpeg is not installed".into());
    }
    if !is_ffprobe_installed() {
        return Err("ffprobe is not installed".into());
    }
    if !Path::new(&args.base_dir).exists() {
        return Err(format!("Directory '{}' does not exist", args.base_dir).into());
    }
    if args.extensions.is_empty() {
        return Err("At least one file extension must be specified".into());
    }
    Ok(())
}