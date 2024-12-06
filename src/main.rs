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
    let (video_infos, total_frames, total_files) = scan_video_files(&args.base_dir, &args.extensions)?;
    
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
    let pb = ProgressBar::new(total_frames as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames ({percent}%) {msg}")
        .unwrap()
        .progress_chars("#>-"));


    // Counters
    let successful_conversions = Arc::new(AtomicUsize::new(0));
    let successful_frames = Arc::new(AtomicUsize::new(0));
    let deleted_files = Arc::new(AtomicUsize::new(0));
    let should_delete = args.delete_originals;


    // Process files in parallel
    video_infos.par_iter()
        .for_each(|video| {


        // Skip if already completed
        if let Some(ref completed) = completed_files {
            if completed.is_completed(&video.path) {
                pb.println(format!("Skipping already encoded file: {}", video.path));
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
	    args.vf.as_deref(),
	    args.copy_audio,
	    args.audio_bitrate,
	    &pb,
	    successful_frames.clone()
	) { 
                Ok(_) => {
                    successful_conversions.fetch_add(1, Ordering::Relaxed);
                    // Log successful conversion and handle deletion
                    match &completed_files {
                        Some(completed) => {
                            if let Err(e) = completed.mark_completed(&video.path) {
                                eprintln!("Failed to log completed file {}: {}", video.path, e);
                            }
                            
                            // Only delete if the file isn't already in the completed log
                            if should_delete && !completed.is_completed(&video.path) {
                                match fs::remove_file(&video.path) {
                                    Ok(_) => {
                                        deleted_files.fetch_add(1, Ordering::Relaxed);
                                        pb.set_message(format!("Converted and deleted: {}", video.path));
                                    },
                                    Err(e) => {
                                        eprintln!("Failed to delete {}: {}", video.path, e);
                                        pb.set_message(format!("Converted (delete failed): {}", video.path));
                                    }
                                }
                            } else {
                                pb.set_message(format!("Converted: {}", video.path));
                            }
                        },
                        None => {
                            // No logging enabled, proceed with normal deletion
                            if should_delete {
                                match fs::remove_file(&video.path) {
                                    Ok(_) => {
                                        deleted_files.fetch_add(1, Ordering::Relaxed);
                                        pb.set_message(format!("Converted and deleted: {}", video.path));
                                    },
                                    Err(e) => {
                                        eprintln!("Failed to delete {}: {}", video.path, e);
                                        pb.set_message(format!("Converted (delete failed): {}", video.path));
                                    }
                                }
                            } else {
                                pb.set_message(format!("Converted: {}", video.path));
                            }
                        }
                    }
                
                },
                Err(e) => {
                    eprintln!("Failed to convert {}: {}", video.path, e);
                }
            }
        });
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
    vf: Option<&str>,
    copy_audio: bool,
    audio_bitrate: Option<u32>,
    pb: &ProgressBar,
    frame_counter: Arc<AtomicUsize>
) -> Result<(), Box<dyn Error>> {
    let input_path = Path::new(input_path);
    let input_stem = input_path.file_stem().unwrap().to_string_lossy();
    
    // Always encode to a temporary file first
    let temp_filename = format!("{}-temp-{}.mkv", input_stem, std::process::id());
    let temp_path = match output_dir {
        Some(dir) => Path::new(dir).join(&temp_filename),
        None => input_path.with_file_name(&temp_filename),
    };

    let audio_channels = get_audio_channels(input_path)?;
    
    let mut cmd = Command::new("ffmpeg");

    if let Some(custom_cmd) = ffmpeg_command {
        // Parse the custom command, keeping -i input and output.mkv handling
        cmd.arg("-i")
            .arg(input_path)
            .args(custom_cmd.split_whitespace())
            .arg("-progress")
            .arg("pipe:1")
            .arg("-y")
            .arg(&temp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
    } else {
        cmd.arg("-i")
            .arg(input_path)
            .arg("-map_metadata")  // Copy all metadata from input
            .arg("0")
            .arg("-c:v")
            .arg("libsvtav1")
            .arg("-preset")
            .arg(preset.to_string())
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-svtav1-params")
            .arg(format!("tune={}:film-grain={}", tune, film_grain))
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        // Add video filter if specified
        if let Some(filter) = vf {
            cmd.arg("-vf").arg(filter);
        }
        if copy_audio {
            cmd.arg("-c:a").arg("copy");
        } else {
            cmd.arg("-c:a").arg("libopus");
            
            // Set appropriate audio parameters based on channel count
            match audio_channels {
                1 => {
                    // Mono
                    cmd.arg("-ac").arg("1")
                       .arg("-b:a").arg(format!("{}k", audio_bitrate.unwrap_or(64)));
                },
                2 => {
                    // Stereo
                    cmd.arg("-ac").arg("2")
                       .arg("-b:a").arg(format!("{}k", audio_bitrate.unwrap_or(96)));
                },
                channels if channels >= 6 => {
                    // 5.1 or higher
                    cmd.arg("-ac").arg("6")
                       .arg("-b:a").arg(format!("{}k", audio_bitrate.unwrap_or(384)))
                       .arg("-mapping_family").arg("1")  // Important for multichannel Opus
                       .arg("-apply_phase_inv").arg("0")
                       .arg("-channel_layout").arg("5.1");
                },
                _ => {
                    // Default to stereo for other configurations
                    cmd.arg("-ac").arg("2")
                       .arg("-b:a").arg(format!("{}k", audio_bitrate.unwrap_or(96)));
                },
            }   
        }

        cmd.arg("-progress")
            .arg("pipe:1")
            .arg("-y")
            .arg(&temp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
    }

    let mut process = cmd.spawn()?;

    let frame_regex = Regex::new(r"frame=\s*(\d+)")?;
    let stdout = process.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    let mut last_frame = 0;
    for line in reader.lines() {
        if let Ok(line) = line {
            if let Some(caps) = frame_regex.captures(&line) {
                if let Some(frame_str) = caps.get(1) {
                    if let Ok(current_frame) = frame_str.as_str().parse::<usize>() {
                        let frame_diff = current_frame - last_frame;
                        frame_counter.fetch_add(frame_diff, Ordering::Relaxed);
                        pb.inc(frame_diff as u64);
                        last_frame = current_frame;
                    }
                }
            }
        }
    }

    let status = process.wait()?;
    if !status.success() {
        // Clean up temp file if encoding failed
        let _ = fs::remove_file(&temp_path);
        return Err("FFmpeg encoding failed".into());
    }

    // Determine final path
    let final_filename = if input_path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("mkv")) {
        input_path.file_name().unwrap().to_string_lossy().to_string()
    } else {
        format!("{}.mkv", input_stem)
    };
    
    let final_path = match output_dir {
        Some(dir) => Path::new(dir).join(&final_filename),
        None => input_path.with_file_name(&final_filename),
    };

    // If we're replacing the original file, we can safely rename now
    if final_path.exists() {
        fs::remove_file(&final_path)?;
    }
    fs::rename(&temp_path, &final_path)?;

    Ok(())
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


fn scan_video_files(base_dir: &str, extensions: &[String]) -> Result<(Vec<VideoInfo>, usize, usize), Box<dyn Error>> {
    let frame_scanning_pb = ProgressBar::new_spinner();
    frame_scanning_pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} Scanning: {msg} ({pos} files)")
        .unwrap());
    frame_scanning_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut video_infos = Vec::new();
    let mut total_files = 0;
    let mut total_frames = 0;
    let mut failed_files = 0;
    
    // Convert all extensions to lowercase once
    let extensions: Vec<String> = extensions.iter()
        .map(|ext| ext.trim().to_lowercase())
        .collect();

    for entry in WalkDir::new(base_dir) {
        if let Ok(entry) = entry {
            if let Some(ext) = entry.path().extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if extensions.contains(&ext_lower) {
                    total_files += 1;
                    frame_scanning_pb.inc(1);
                    frame_scanning_pb.set_message(entry.path().display().to_string());

                    match get_frame_count(entry.path()) {
                        Ok(frame_count) => {
                            total_frames += frame_count;
                            video_infos.push(VideoInfo {
                                path: entry.path().to_string_lossy().into_owned(),
                            });
                        }
                        Err(e) => {
                            failed_files += 1;
                            eprintln!("Skipping {}: ffprobe failed: {}", entry.path().display(), e);
                            continue;
                        }
                    }
                }
            }
        }
    }

    frame_scanning_pb.finish_with_message(format!(
        "Scanned {} files ({} skipped due to ffprobe failures)", 
        total_files,
        failed_files
    ));
    
    Ok((video_infos, total_frames, total_files))
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