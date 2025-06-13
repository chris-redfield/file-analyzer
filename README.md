# AI File Processor

An file processing system that uses GenAI models to automatically analyze, tag, and categorize files on your system. Supports both local processing with Ollama and cloud processing with OpenAI.

## Features

- 🔍 **Automated File Discovery**: Uses system commands to efficiently discover all accessible files
- 🏷️ **Smart Tagging**: AI-powered content analysis and tagging for text and image files
- 🔐 **Encryption Detection**: Automatically identifies encrypted files using entropy analysis
- 📍 **Location Recognition**: Extracts location information from both text and images
- 👥 **Entity Extraction**: Identifies people, organizations, and other entities
- 🚀 **Batch Processing**: Efficient batch processing with configurable batch sizes
- 📊 **Comprehensive Reporting**: Detailed JSON and CSV reports of processing results
- 🤖 **Dual AI Support**: Works with both Ollama (local) and OpenAI (cloud)

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd ai-file-processor
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure**:
   ```bash
   cp .env.template .env
   # Edit .env with your preferences
   ```

3. **Run**:
   ```bash
   python3 ai_file_processor.py
   ```

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_OLLAMA` | `true` | Use Ollama (local) vs OpenAI (cloud) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3` | Ollama model for processing |
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `MAX_FILE_SIZE` | `52428800` | Maximum file size to process (50MB) |
| `BATCH_SIZE` | `5` | Number of files to process per batch |
| `OUTPUT_DIR` | `ai_processing_results` | Directory for results |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Priority Folders

The system processes files from these folders first:
- `~/Downloads/`
- `~/Documents/`
- `~/Pictures/`

You can customize this in the .env file with `PRIORITY_FOLDERS`.

## Supported File Types

### Text Files
- `.txt`, `.py`, `.js`, `.html`, `.css`, `.md`
- `.json`, `.xml`, `.yml`, `.yaml`, `.csv`
- Any file with text MIME type

### Image Files
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`
- `.tiff`, `.webp`, `.svg`
- Any file with image MIME type

### Encrypted Files (Detected but not processed)
- Files with high entropy (>7.5 Shannon entropy)
- Common encrypted extensions (`.gpg`, `.pgp`, `.enc`, `.aes`)
- Password-protected archives

## Output

The system generates several output files in the configured output directory:

### `processing_results.json`
Complete processing results including:
- Processing metadata and configuration
- Summary statistics
- Detailed results for each file processed

### `summary.csv`
Spreadsheet-friendly summary with columns:
- File path and basic metadata
- Processing status and results
- Tags, entities, and locations found
- Any errors encountered

### `processing.log`
Detailed processing log with timestamps and debug information.

## Usage Examples

### Basic Usage (Ollama)
```bash
# Ensure Ollama is running
ollama serve

# Run with default settings
python3 ai_file_processor.py
```

### OpenAI Mode
```bash
# Set environment variables
export USE_OLLAMA=false
export OPENAI_API_KEY=your_api_key_here

# Or edit .env file
python3 ai_file_processor.py
```

### Custom Configuration
```bash
# Edit .env for custom settings
nano .env

# Run with custom batch size and output directory
export BATCH_SIZE=10
export OUTPUT_DIR=my_results
python3 ai_file_processor.py
```

## AI Model Requirements

### Ollama (Recommended for Privacy)
- **Model**: `gemma3` (supports both text and image processing)
- **Installation**: Automatic via setup script
- **Benefits**: Local processing, no API costs, privacy-friendly

### OpenAI
- **Model**: `gpt-4o` (supports vision and text)
- **Requirements**: Valid OpenAI API key
- **Benefits**: Higher accuracy, faster processing, no local resources needed

## Performance and Limitations

### File Size Limits
- Default maximum: 50MB per file
- Configurable via `MAX_FILE_SIZE`
- Large files are skipped to prevent memory issues

### Processing Speed
- **Ollama**: ~2-5 seconds per file (depends on hardware)
- **OpenAI**: ~1-3 seconds per file (depends on API limits)
- **Batch processing**: Optimized for efficiency

### System Requirements
- **GPU RAM**: 8GB+ recommended but depends on model
- **Storage**: Varies based on model size

## Troubleshooting

### Common Issues

1. **Ollama server not starting**:
   ```bash
   # Check if port is available
   netstat -an | grep 11434
   
   # Start manually
   ollama serve
   ```

2. **Permission denied errors**:
   - Normal for system directories
   - Check logs for accessible file count

3. **Out of memory**:
   - Reduce `BATCH_SIZE`
   - Increase `MAX_FILE_SIZE` limit
   - Close other applications

4. **Model not found**:
   ```bash
   # Pull required model
   ollama pull gemma3
   ```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python3 ai_file_processor.py
```

## Security Considerations

- **File Access**: Only reads files your user account can access
- **Encrypted Files**: Detected but not processed for security
- **API Keys**: Store securely in .env file (never commit to version control)
- **Local Processing**: Ollama keeps data on your machine
- **Network**: OpenAI mode sends file content to OpenAI servers

## Advanced Usage

### Custom Priority Folders
```bash
export PRIORITY_FOLDERS="~/Desktop/,~/Projects/,~/Important/"
```

### Processing Specific Directories
Modify the `_manual_file_discovery()` method to target specific directories.

### Custom AI Prompts
Edit the prompt templates in `_process_text_file()` and `_process_image_file()` methods.
