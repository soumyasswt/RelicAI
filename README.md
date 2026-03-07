# RelicAI - AI Research Engine

A powerful AI research engine built with React and Vite, featuring local Ollama integration for zero-cost AI inference.

## Features

- **Local AI Inference**: Uses Ollama with Llama 3 for completely free AI processing
- **Vector Search**: BM25 + TF-IDF hybrid search engine for document retrieval
- **Cyberpunk Theme**: Techy, hacker-style UI with neon accents
- **Real-time Streaming**: Live response streaming from local AI models
- **Persistent Storage**: Local document storage and retrieval

## Prerequisites

1. **Node.js** (v16 or higher)
2. **Ollama** installed and running locally
3. **Llama 3 model** pulled: `ollama pull llama3`

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   npm install
   ```

## Usage

1. Start Ollama service (if not already running)
2. Pull the Llama 3 model:
   ```bash
   ollama pull llama3
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser to `http://localhost:5173`

## Architecture

- **Frontend**: React + Vite
- **AI Engine**: Local Ollama (Llama 3)
- **Search**: Custom BM25 + TF-IDF vector engine
- **Storage**: Browser local storage
- **Streaming**: NDJSON parsing for real-time responses

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Troubleshooting

- **Infinite loading**: Ensure Ollama is running and Llama 3 model is pulled
- **API errors**: Check that Ollama service is accessible at `http://localhost:11434`
- **Build errors**: Make sure all dependencies are installed with `npm install`