"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Sparkles,
  CloudRain,
  Leaf,
  BarChart3,
  Zap,
  Menu,
  Plus,
  MessageSquare,
  Trash2,
  X,
  Cpu,
  Globe
} from "lucide-react";
import VoiceAssistant from "@/components/VoiceAssistant";
import ChatMessage from "@/components/ChatMessage";
import { cn } from "@/lib/utils";

// --- Types ---
interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  structuredData?: any;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  timestamp: number;
}

const BACKEND_URL = "/api";

const SAMPLE_QUERIES = [
  { icon: CloudRain, text: "Annual rainfall for Davangere", color: "text-blue-500 bg-blue-50" },
  { icon: Leaf, text: "Top 5 crops in Davangere 2015", color: "text-green-500 bg-green-50" },
  { icon: BarChart3, text: "Rice production trends", color: "text-purple-500 bg-purple-50" },
  { icon: Zap, text: "Optimal rainfall for wheat", color: "text-orange-500 bg-orange-50" },
];

export default function Home() {
  // --- State ---
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true); // Mobile toggle
  const [showAbout, setShowAbout] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // --- Effects ---

  // Load from LocalStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("samarth_chats");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setChats(parsed);
        if (parsed.length > 0) setCurrentChatId(parsed[0].id);
        else createNewChat();
      } catch (e) {
        createNewChat();
      }
    } else {
      createNewChat();
    }
  }, []);

  // Save to LocalStorage whenever chats change
  useEffect(() => {
    if (chats.length > 0) {
      localStorage.setItem("samarth_chats", JSON.stringify(chats));
    }
  }, [chats]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, currentChatId, isLoading]);

  // --- Helpers ---

  const generateId = () => Math.random().toString(36).substring(2, 15);

  const getCurrentChat = () => chats.find((c) => c.id === currentChatId) || chats[0];

  const createNewChat = () => {
    const newChat: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [],
      timestamp: Date.now(),
    };
    setChats((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    if (window.innerWidth < 768) setSidebarOpen(false); // Close sidebar on mobile
  };

  const deleteChat = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newChats = chats.filter((c) => c.id !== id);
    setChats(newChats);
    if (currentChatId === id) {
      if (newChats.length > 0) setCurrentChatId(newChats[0].id);
      else createNewChat();
    }
  };

  const updateCurrentChatMessages = (newMessages: Message[]) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id === currentChatId ? { ...c, messages: newMessages } : c
      )
    );
  };

  const updateChatTitle = (id: string, firstMessage: string) => {
    setChats(prev => prev.map(c => {
      if (c.id === id && c.title === "New Chat") {
        return { ...c, title: firstMessage.slice(0, 30) + (firstMessage.length > 30 ? "..." : "") };
      }
      return c;
    }));
  }

  // --- Handlers ---

  const regenerateResponse = async () => {
    const currentChat = getCurrentChat();
    // Must have at least one user message
    if (!currentChat || currentChat.messages.length === 0 || isLoading) return;

    let newMessages = [...currentChat.messages];
    const lastMsg = newMessages[newMessages.length - 1];

    // If last was assistant, remove it to regenerate
    if (lastMsg.role === "assistant") {
      newMessages.pop();
    }

    // Now last must be user
    const lastUserMsg = newMessages[newMessages.length - 1];
    if (!lastUserMsg || lastUserMsg.role !== "user") return;

    // Update state to remove old AI response immediately
    updateCurrentChatMessages(newMessages);
    setIsLoading(true);

    try {
      const history = newMessages.map(m => ({ role: m.role, content: m.content }));

      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: lastUserMsg.content, // Use last user message prompt
          session_id: currentChatId,
          history: history
        }),
      });

      if (!response.ok) throw new Error("Failed to connect");
      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        structuredData: data.structured_data,
      };

      // Append new response
      updateCurrentChatMessages([...newMessages, assistantMessage]);

    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content: `System Error: ${error}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      updateCurrentChatMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };


  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading || !currentChatId) return;

    const currentChat = getCurrentChat();
    const userMessage: Message = {
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    };

    const updatedMessages = [...currentChat.messages, userMessage];
    updateCurrentChatMessages(updatedMessages);
    updateChatTitle(currentChatId, text);

    setInput("");
    setIsLoading(true);

    try {
      // Prepare history for backend (map to role/content)
      const history = updatedMessages.map(m => ({ role: m.role, content: m.content }));

      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: currentChatId,
          history: history // Send persistent history
        }),
      });

      if (!response.ok) throw new Error("Failed to connect");
      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        structuredData: data.structured_data,
      };

      updateCurrentChatMessages([...updatedMessages, assistantMessage]);

    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content: `System Error: ${error}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      updateCurrentChatMessages([...updatedMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const currentChat = getCurrentChat();

  return (
    <div className="flex h-screen bg-white text-gray-900 font-sans overflow-hidden">

      {/* --- Sidebar --- */}
      <AnimatePresence mode="wait">
        {(sidebarOpen || (typeof window !== 'undefined' && window.innerWidth >= 768)) && (
          <motion.aside
            initial={{ x: -260, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -260, opacity: 0 }}
            className="fixed md:relative z-50 w-[260px] h-full bg-gray-900 text-gray-100 flex flex-col shrink-0 border-r border-gray-800"
          >
            {/* Logo Area */}
            <div className="p-4 flex items-center justify-between">
              <div onClick={() => setShowAbout(true)} className="flex items-center gap-2 cursor-pointer hover:bg-gray-800 p-2 rounded-lg transition-colors">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <span className="font-semibold text-lg tracking-tight">Samarth AI</span>
              </div>
              {/* Mobile Close */}
              <button onClick={() => setSidebarOpen(false)} className="md:hidden p-2 text-gray-400">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* New Chat Button */}
            <div className="px-3 mb-2">
              <button
                onClick={createNewChat}
                className="flex items-center gap-2 w-full px-3 py-3 rounded-lg border border-gray-700 hover:bg-gray-800 transition-colors text-sm font-medium text-left"
              >
                <Plus className="w-4 h-4" />
                New chat
              </button>
            </div>

            {/* Chat List */}
            <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-hide">
              <div className="text-xs font-semibold text-gray-500 mb-2 px-2 uppercase tracking-wider">Recent</div>
              {chats.map(chat => (
                <div
                  key={chat.id}
                  onClick={() => { setCurrentChatId(chat.id); if (window.innerWidth < 768) setSidebarOpen(false); }}
                  className={cn(
                    "group flex items-center gap-3 px-3 py-3 text-sm rounded-lg cursor-pointer transition-colors relative",
                    chat.id === currentChatId ? "bg-gray-800 text-white" : "text-gray-300 hover:bg-gray-800/50"
                  )}
                >
                  <MessageSquare className="w-4 h-4 shrink-0 text-gray-500 group-hover:text-gray-300" />
                  <span className="truncate flex-1">{chat.title}</span>

                  {/* Delete Button (Visible on Hover/Active) */}
                  {chats.length > 1 && (
                    <button
                      onClick={(e) => deleteChat(e, chat.id)}
                      className={cn("opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity", chat.id === currentChatId && "opacity-100")}
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
            </div>

          </motion.aside>
        )}
      </AnimatePresence>

      {/* --- Main Area --- */}
      <main className="flex-1 flex flex-col h-full relative w-full overflow-hidden">

        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between p-4 border-b border-gray-100 bg-white">
          <button onClick={() => setSidebarOpen(true)} className="p-2 -ml-2 text-gray-600">
            <Menu className="w-6 h-6" />
          </button>
          <span className="font-semibold text-gray-800">Samarth AI</span>
          <div className="w-8" />
        </header>

        {/* Messages Info Area */}
        <div className="flex-1 overflow-y-auto scroll-smooth pb-40">
          {currentChat && currentChat.messages.length === 0 ? (
            // Empty State
            <div className="h-full flex flex-col items-center justify-center p-8 text-center space-y-8 min-h-[500px]">
              <div className="w-16 h-16 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-blue-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-800">How can I help you today?</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
                {SAMPLE_QUERIES.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => sendMessage(q.text)}
                    className="p-4 rounded-xl border border-gray-200 hover:bg-gray-50 transition-colors text-left flex items-start gap-3 group"
                  >
                    <q.icon className={cn("w-5 h-5 shrink-0 mt-0.5", q.color.split(" ")[0])} />
                    <div>
                      <p className="text-sm font-medium text-gray-700">{q.text}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            // Messages
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {currentChat?.messages.map((msg, i) => (
                <ChatMessage key={i} {...msg} />
              ))}
              {isLoading && (
                <div className="flex gap-4 w-full pl-0 md:pl-0">
                  <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shrink-0 animate-pulse">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <div className="space-y-2 w-full pt-1 max-w-[80%]">
                    <div className="h-4 bg-gray-200 rounded w-1/4 animate-pulse"></div>
                    <div className="h-4 bg-gray-100 rounded w-3/4 animate-pulse"></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area (Bottom) - Mobile Optimized */}
        <div className="w-full bg-white/95 backdrop-blur-sm border-t border-gray-100 p-2 sm:p-4 absolute bottom-0 left-0 safe-area-bottom">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={handleSubmit} className="relative">
              <input
                ref={inputRef}
                className="w-full bg-gray-50 border border-gray-200 rounded-xl pl-3 sm:pl-4 pr-20 sm:pr-28 py-3 sm:py-4 text-sm sm:text-base focus:ring-2 focus:ring-blue-100 focus:border-blue-300 outline-none transition-all shadow-sm"
                placeholder="Message Samarth AI..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
              />
              <div className="absolute right-1.5 sm:right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5 sm:gap-1">
                <VoiceAssistant
                  onTranscript={(t) => setInput(t)}
                  isListening={isListening}
                  setIsListening={setIsListening}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className={cn(
                    "p-2 sm:p-2.5 rounded-lg transition-all duration-200",
                    input.trim() ? "bg-blue-600 text-white hover:bg-blue-700 shadow-md" : "bg-gray-200 text-gray-400"
                  )}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </form>
            <p className="hidden sm:block text-center text-xs text-gray-400 mt-2">
              Samarth AI can make mistakes. Consider checking important information.
            </p>
          </div>
        </div>

      </main>

      {/* About Modal */}
      <AnimatePresence>
        {showAbout && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4">
            <div className="bg-white rounded-2xl p-6 max-w-md w-full relative shadow-xl">
              <button onClick={() => setShowAbout(false)} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"><X className="w-5 h-5" /></button>
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <Sparkles className="w-6 h-6" />
                </div>
                <h2 className="text-xl font-bold mb-2">Samarth AI</h2>
                <p className="text-gray-600 text-sm">
                  Advanced agricultural intelligence powered by Groq Llama 3.3.
                  Designed for professionals.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
