# Samarth AI - Advanced Agricultural Intelligence Platform

## 🚀 Features

- **Voice Assistant**: Speak your questions and get voice responses
- **Advanced AI Chat**: Intelligent responses with structured data visualization
- **Real-time Analytics**: Interactive charts and tables for agricultural data
- **Beautiful UI**: Modern, light-themed interface with smooth animations
- **Responsive Design**: Works perfectly on all devices

## 🏗️ Architecture

### Backend (Python/FastAPI)
- Location: Root directory
- Entry: `api.py`
- Agent Logic: `agent/core.py`
- Database: `data/processed/samarth_data.db`

### Frontend (Next.js/React)
- Location: `frontend/`
- Framework: Next.js 15 with TypeScript
- Styling: Tailwind CSS
- Animations: Framer Motion
- Voice: Web Speech API

## 📦 Installation

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
# Create .env file with:
GROQ_API_KEY=your_groq_api_key_here

# Run backend
python api.py
# Or with uvicorn:
uvicorn api:app --reload
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## 🌐 Deployment

### Backend (Render)
1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Render will auto-detect `render.yaml`
5. Add `GROQ_API_KEY` in environment variables
6. Deploy!

### Frontend (Vercel)
1. Push code to GitHub
2. Import project in Vercel
3. Set root directory to `frontend`
4. Add environment variable:
   - `NEXT_PUBLIC_BACKEND_URL`: Your Render backend URL
5. Deploy!

## 🎯 Usage

### Text Input
Type your questions in the chat input:
- "Annual rainfall for Davangere district"
- "Top 5 crops in Davangere in 2015"
- "Compare Rice production in Karnataka 2010 vs 2020"

### Voice Input
1. Click the microphone button
2. Speak your question
3. Get voice response automatically

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- LangChain
- Groq LLM
- SQLite/DuckDB
- Pandas

**Frontend:**
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Framer Motion
- Lucide Icons
- Web Speech API

## 📝 Environment Variables

### Backend (.env)
```
GROQ_API_KEY=your_api_key
```

### Frontend
Set in Vercel dashboard or create `.env.local`:
```
NEXT_PUBLIC_BACKEND_URL=https://your-backend.onrender.com
```

## 🎨 Features Implemented

✅ Advanced voice assistant with speech recognition
✅ Text-to-speech for AI responses
✅ Beautiful light-themed UI
✅ Animated components with Framer Motion
✅ Structured data visualization
✅ Interactive charts for trends
✅ Responsive design
✅ Real-time chat interface
✅ Sample query suggestions
✅ Loading states and animations

## 👨‍💻 Developer

Created by **Vashista C V**

