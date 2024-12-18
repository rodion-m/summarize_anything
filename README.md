# Summarize Anything

A powerful Python-based tool that downloads YouTube videos, transcribes their audio using the DeepInfra API, summarizes the content, and generates both HTML and PDF summaries. It also supports processing existing SRT files for summarization.

![Summarize Anything](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Transcribe and Summarize a YouTube Video](#transcribe-and-summarize-a-youtube-video)
  - [Process an Existing SRT File](#process-an-existing-srt-file)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Download YouTube Audio**: Extracts audio from YouTube videos using `yt-dlp`.
- **Transcription**: Transcribes audio using the DeepInfra Whisper API with support for chunked processing.
- **Summarization**: Generates detailed summaries of the transcribed content in HTML format.
- **PDF Generation**: Converts HTML summaries to PDF using WeasyPrint.
- **SRT Processing**: Supports processing existing SRT/VTT subtitle files.
- **Caching**: Caches transcriptions to avoid redundant processing.
- **Logging**: Detailed logging for monitoring and debugging.

## Prerequisites

### 1. macOS Setup

This tool relies on several dependencies, including WeasyPrint for PDF generation. Follow the steps below to set up your macOS environment.

#### a. Install Homebrew

Homebrew is a package manager for macOS. If you don’t have Homebrew installed, open your terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Verify the installation:

```bash
brew --version
```

#### b. Install Required Libraries

WeasyPrint requires several libraries such as Cairo, Pango, GDK-Pixbuf, and GTK+3.

- **For Intel Macs:**

  ```bash
  brew install libffi glib gobject-introspection cairo pango gdk-pixbuf gtk+3
  ```

- **For Apple Silicon (M1/M2 Macs):**

  ```bash
  arch -arm64 brew install libffi glib gobject-introspection cairo pango gdk-pixbuf gtk+3
  ```

#### c. Export Library Paths

Ensure the system can locate the installed libraries by updating your environment variables.

- **For Intel Macs:** Add the following lines to your shell configuration file (e.g., `~/.zshrc` or `~/.bash_profile`):

  ```bash
  export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
  export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
  ```

- **For Apple Silicon (M1/M2 Macs):** Add these lines to your shell configuration file:

  ```bash
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
  export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
  export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
  ```

Reload the configuration:

```bash
source ~/.zshrc  # or source ~/.bash_profile
```

#### d. Verify Installation

Check if the required libraries are available:

```bash
pkg-config --cflags --libs gobject-2.0 cairo pango
```

If no errors are reported, the libraries are correctly installed.

### 2. Python Environment

Ensure you have Python 3.1 or higher installed. You can check your Python version with:

```bash
python3 --version
```

### 3. API Keys

This tool requires API keys for DeepInfra and OpenRouter. Obtain your API keys from their respective platforms and store them securely.

OpenRouter: https://openrouter.ai/settings/keys

DeepInfra: https://deepinfra.com/dash/api_keys

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/youtube-transcription-summarization.git
   cd youtube-transcription-summarization
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, you can install dependencies directly:*

   ```bash
   pip install argparse backoff litellm requests webvtt-py yt-dlp python-dotenv pydantic WeasyPrint
   ```

## Configuration

1. **Environment Variables**

   Create a `.env` file in the project root directory and add your API keys:

   ```env
   DEEPINFRA_API_KEY=your_deepinfra_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

   Replace `your_deepinfra_api_key` and `your_openrouter_api_key` with your actual API keys.
    
DEEPINFRA_API_KEY is required only for audio transcription. 

2. **Verify WeasyPrint Installation**

   Ensure that WeasyPrint can generate PDFs by running a test script or using the provided `test_weasyprint.py` as described in the [WeasyPrint Setup Guide](#prerequisites).

## Usage

The tool can be used to process YouTube videos or existing SRT/VTT files. Below are the instructions for both use cases.

### Transcribe and Summarize a YouTube Video

1. **Basic Command**

   ```bash
   python main.py --youtube "https://www.youtube.com/watch?v=your_video_id" --target-language "English"
   ```

   - `--youtube`: URL of the YouTube video to process.
   - `--target-language`: Target language for the summary (e.g., "English", "Spanish").

2. **Optional Arguments**

   - `--use-subtitles`: Use subtitles from YouTube if available (default: `True`).
   - `--output-dir`: Directory to save output files (default: `output`).

3. **Example**

   ```bash
   python main.py --youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --target-language "Spanish" --output-dir "results"
   ```

### Process an Existing SRT File

1. **Command**

   ```bash
   python main.py --srt "/path/to/subtitles.srt" --target-language "French"
   ```

   - `--srt`: Path to an existing SRT or VTT file.
   - `--target-language`: Target language for the summary.

2. **Example**

   ```bash
   python main.py --srt "./subtitles/video_subtitles.srt" --target-language "German" --output-dir "summaries"
   ```

## Output

The tool generates several output files in the specified `--output-dir` (default is `output`):

- **Transcription Files:**
  - `video_id_transcription.txt`: Full transcription text.
  - `video_id_transcription.srt`: SRT file with timed subtitles.

- **Summary Files:**
  - `video_id_summary.html`: HTML file containing the summarized content.
  - `video_id_summary.pdf`: PDF version of the HTML summary.

- **Example:**

  ```
  output/
  ├── dQw4w9WgXcQ_transcription.txt
  ├── dQw4w9WgXcQ_transcription.srt
  ├── dQw4w9WgXcQ_summary.html
  └── dQw4w9WgXcQ_summary.pdf
  ```

## Troubleshooting

### Common Issues

1. **WeasyPrint PDF Generation Errors**

   - **Error:** `OSError: cannot open resource`
   - **Solution:** Ensure all required libraries (Cairo, Pango, GDK-Pixbuf, GTK+3) are correctly installed and the environment variables are properly set. Revisit the [Prerequisites](#prerequisites) section.

2. **Missing API Keys**

   - **Error:** `NoneType` related to API keys.
   - **Solution:** Ensure your `.env` file contains valid `DEEPINFRA_API_KEY` and `OPENROUTER_API_KEY`.

3. **YouTube Download Failures**

   - **Error:** `FileNotFoundError` or download-related exceptions.
   - **Solution:** Verify the YouTube URL is correct and accessible. Ensure `yt-dlp` is up to date:

     ```bash
     pip install --upgrade yt-dlp
     ```

4. **Transcription Failures**

   - **Error:** API request errors or transcription issues.
   - **Solution:** Check your DeepInfra API key and ensure you have sufficient quota. Review network connectivity and API status.

5. **SRT Parsing Errors**

   - **Error:** Invalid SRT format or parsing exceptions.
   - **Solution:** Ensure the SRT/VTT file is correctly formatted. Use tools like [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit) to validate and fix subtitle files.

### Logging

The tool provides detailed logs to help identify issues. Review the console output for error messages and debugging information.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

---

**Disclaimer:** Ensure you have the rights to download and process YouTube videos. Respect copyright laws and YouTube's terms of service.