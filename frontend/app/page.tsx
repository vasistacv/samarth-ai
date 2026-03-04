"use client";

import { useState, useRef, useEffect, useCallback, KeyboardEvent as ReactKeyboardEvent } from "react";
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
  Search,
  Pin,
  PinOff,
  Moon,
  Sun,
  Command,
  ArrowUp,
  Settings,
  Keyboard,
  Globe,
  Code2,
  TrendingUp,
  Brain,
  Lightbulb,
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
  pinned?: boolean;
}

const BACKEND_URL = "/api";

const SAMPLE_QUERIES = [
  {
    icon: CloudRain,
    text: "Annual rainfall for Davangere district",
    subtitle: "Weather analytics",
    gradient: "from-blue-500/10 to-cyan-500/10",
    iconColor: "text-blue-500",
    borderColor: "hover:border-blue-400/50",
  },
  {
    icon: Leaf,
    text: "Top 5 crops in Davangere 2015",
    subtitle: "Crop intelligence",
    gradient: "from-emerald-500/10 to-green-500/10",
    iconColor: "text-emerald-500",
    borderColor: "hover:border-emerald-400/50",
  },
  {
    icon: BarChart3,
    text: "Compare Rice production trends",
    subtitle: "Production analytics",
    gradient: "from-violet-500/10 to-purple-500/10",
    iconColor: "text-violet-500",
    borderColor: "hover:border-violet-400/50",
  },
  {
    icon: Zap,
    text: "Optimal rainfall for wheat cultivation",
    subtitle: "AI recommendations",
    gradient: "from-amber-500/10 to-orange-500/10",
    iconColor: "text-amber-500",
    borderColor: "hover:border-amber-400/50",
  },
];

const CAPABILITY_CARDS = [
  { icon: Brain, label: "Agricultural Intelligence", desc: "Powered by Llama 3.3" },
  { icon: TrendingUp, label: "Data Analytics", desc: "Real-time insights" },
  { icon: Code2, label: "Code Generation", desc: "Full-stack solutions" },
  { icon: Globe, label: "Knowledge Base", desc: "Multi-domain expertise" },
];

export default function Home() {
  // --- State ---
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showAbout, setShowAbout] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // --- Theme ---
  useEffect(() => {
    const saved = localStorage.getItem("samarth_theme");
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("samarth_theme", next);
  };

  // --- Load chats ---
  useEffect(() => {
    const saved = localStorage.getItem("samarth_chats");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setChats(parsed);
        if (parsed.length > 0) setCurrentChatId(parsed[0].id);
        else createNewChat();
      } catch {
        createNewChat();
      }
    } else {
      createNewChat();
    }
  }, []);

  // --- Save chats ---
  useEffect(() => {
    if (chats.length > 0) {
      localStorage.setItem("samarth_chats", JSON.stringify(chats));
    }
  }, [chats]);

  // --- Scroll to bottom ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, currentChatId, isLoading]);

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handler = (e: globalThis.KeyboardEvent) => {
      // Ctrl+N or Cmd+N = New chat
      if ((e.ctrlKey || e.metaKey) && e.key === "n") {
        e.preventDefault();
        createNewChat();
      }
      // Ctrl+/ = Focus input
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        textareaRef.current?.focus();
      }
      // Ctrl+K = Search chats
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setShowSearch((p) => !p);
      }
      // Ctrl+Shift+? = Show shortcuts
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "?") {
        e.preventDefault();
        setShowShortcuts((p) => !p);
      }
      // Escape = Close modals
      if (e.key === "Escape") {
        setShowAbout(false);
        setShowSearch(false);
        setShowShortcuts(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // --- Auto-resize textarea ---
  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input, adjustTextareaHeight]);

  // --- Helpers ---
  const generateId = () => Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

  const getCurrentChat = () => chats.find((c) => c.id === currentChatId) || chats[0];

  const createNewChat = () => {
    const newChat: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [],
      timestamp: Date.now(),
      pinned: false,
    };
    setChats((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    setInput("");
    if (typeof window !== "undefined" && window.innerWidth < 768) setSidebarOpen(false);
    setTimeout(() => textareaRef.current?.focus(), 100);
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

  const togglePin = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setChats((prev) =>
      prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c))
    );
  };

  const updateCurrentChatMessages = (newMessages: Message[]) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id === currentChatId ? { ...c, messages: newMessages } : c
      )
    );
  };

  const updateChatTitle = (id: string, firstMessage: string) => {
    setChats((prev) =>
      prev.map((c) => {
        if (c.id === id && c.title === "New Chat") {
          return {
            ...c,
            title: firstMessage.slice(0, 35) + (firstMessage.length > 35 ? "..." : ""),
          };
        }
        return c;
      })
    );
  };

  // --- Handlers ---
  const regenerateResponse = async () => {
    const currentChat = getCurrentChat();
    if (!currentChat || currentChat.messages.length === 0 || isLoading) return;

    let newMessages = [...currentChat.messages];
    const lastMsg = newMessages[newMessages.length - 1];
    if (lastMsg.role === "assistant") newMessages.pop();

    const lastUserMsg = newMessages[newMessages.length - 1];
    if (!lastUserMsg || lastUserMsg.role !== "user") return;

    updateCurrentChatMessages(newMessages);
    setIsLoading(true);

    try {
      const history = newMessages.map((m) => ({ role: m.role, content: m.content }));
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: lastUserMsg.content, session_id: currentChatId, history }),
      });

      if (!response.ok) throw new Error("Failed to connect");
      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        structuredData: data.structured_data,
      };

      updateCurrentChatMessages([...newMessages, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content: `⚠️ Connection error. Please try again.`,
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

    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      const history = updatedMessages.map((m) => ({ role: m.role, content: m.content }));
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: currentChatId, history }),
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
        content: `⚠️ Connection error. Please check your connection and try again.`,
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

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const currentChat = getCurrentChat();

  // --- Filter chats by search ---
  const filteredChats = chats.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const pinnedChats = filteredChats.filter((c) => c.pinned);
  const unpinnedChats = filteredChats.filter((c) => !c.pinned);

  return (
    <div
      className="flex h-screen overflow-hidden transition-colors duration-300"
      style={{ background: "var(--bg-primary)", color: "var(--text-primary)" }}
    >
      {/* ===================== SIDEBAR ===================== */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <>
            {/* Mobile overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 z-40 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
            <motion.aside
              initial={{ x: -280, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -280, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="sidebar fixed md:relative z-50 w-[280px] h-full flex flex-col shrink-0"
            >
              {/* Sidebar Header */}
              <div className="p-4 flex items-center justify-between">
                <button
                  onClick={() => setShowAbout(true)}
                  className="flex items-center gap-3 hover:opacity-80 transition-opacity"
                >
                  <div className="w-9 h-9 gradient-bg rounded-xl flex items-center justify-center shadow-lg">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <span className="font-semibold text-[15px] tracking-tight text-white">
                      Samarth AI
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      <span className="text-[10px] text-gray-500 font-medium">Online</span>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-white/5 transition-all md:hidden"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* New Chat Button */}
              <div className="px-3 mb-3">
                <button
                  onClick={createNewChat}
                  className="flex items-center gap-2.5 w-full px-4 py-3 rounded-xl border border-gray-700/60 hover:bg-white/5 transition-all text-sm font-medium text-gray-300 group"
                >
                  <Plus className="w-4 h-4 text-gray-500 group-hover:text-white transition-colors" />
                  <span>New chat</span>
                  <div className="ml-auto flex items-center gap-0.5">
                    <span className="kbd text-gray-600">⌘</span>
                    <span className="kbd text-gray-600">N</span>
                  </div>
                </button>
              </div>

              {/* Search */}
              <div className="px-3 mb-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
                  <input
                    type="text"
                    placeholder="Search chats..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-white/5 border border-gray-800 rounded-lg pl-9 pr-3 py-2 text-xs text-gray-300 placeholder:text-gray-600 outline-none focus:border-gray-600 transition-colors"
                  />
                </div>
              </div>

              {/* Chat List */}
              <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5 scrollbar-hide">
                {/* Pinned Section */}
                {pinnedChats.length > 0 && (
                  <div className="mb-3">
                    <div className="text-[10px] font-semibold text-gray-600 mb-1.5 px-2 uppercase tracking-[0.1em] flex items-center gap-1.5">
                      <Pin className="w-3 h-3" />
                      Pinned
                    </div>
                    {pinnedChats.map((chat) => (
                      <SidebarChatItem
                        key={chat.id}
                        chat={chat}
                        isActive={chat.id === currentChatId}
                        onSelect={() => {
                          setCurrentChatId(chat.id);
                          if (typeof window !== "undefined" && window.innerWidth < 768) setSidebarOpen(false);
                        }}
                        onDelete={(e) => deleteChat(e, chat.id)}
                        onTogglePin={(e) => togglePin(e, chat.id)}
                        canDelete={chats.length > 1}
                      />
                    ))}
                  </div>
                )}

                {/* Recent Section */}
                <div>
                  <div className="text-[10px] font-semibold text-gray-600 mb-1.5 px-2 uppercase tracking-[0.1em]">
                    Recent
                  </div>
                  {unpinnedChats.map((chat) => (
                    <SidebarChatItem
                      key={chat.id}
                      chat={chat}
                      isActive={chat.id === currentChatId}
                      onSelect={() => {
                        setCurrentChatId(chat.id);
                        if (typeof window !== "undefined" && window.innerWidth < 768) setSidebarOpen(false);
                      }}
                      onDelete={(e) => deleteChat(e, chat.id)}
                      onTogglePin={(e) => togglePin(e, chat.id)}
                      canDelete={chats.length > 1}
                    />
                  ))}
                </div>
              </div>

              {/* Sidebar Footer */}
              <div className="p-3 border-t border-gray-800/80 space-y-2">
                {/* Theme Toggle */}
                <button
                  onClick={toggleTheme}
                  className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-all"
                >
                  {theme === "dark" ? (
                    <Sun className="w-4 h-4" />
                  ) : (
                    <Moon className="w-4 h-4" />
                  )}
                  <span className="text-[13px]">{theme === "dark" ? "Light mode" : "Dark mode"}</span>
                </button>

                {/* Shortcuts */}
                <button
                  onClick={() => setShowShortcuts(true)}
                  className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-all"
                >
                  <Keyboard className="w-4 h-4" />
                  <span className="text-[13px]">Keyboard shortcuts</span>
                  <span className="ml-auto kbd text-gray-600">?</span>
                </button>

                {/* Developer Credit */}
                <div className="px-3 pt-2 pb-1 text-center">
                  <p className="text-[10px] text-gray-600 font-medium">
                    Developed by{" "}
                    <span className="text-gray-400 font-semibold">Vashista C V</span>
                  </p>
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* ===================== MAIN AREA ===================== */}
      <main className="flex-1 flex flex-col h-full relative w-full overflow-hidden">
        {/* Top Bar */}
        <header
          className="flex items-center justify-between px-4 py-3 shrink-0"
          style={{
            borderBottom: "1px solid var(--border-primary)",
            background: "var(--bg-primary)",
          }}
        >
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
                style={{ color: "var(--text-secondary)" }}
              >
                <Menu className="w-5 h-5" />
              </button>
            )}
            {!sidebarOpen && (
              <button
                onClick={createNewChat}
                className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
                style={{ color: "var(--text-secondary)" }}
              >
                <Plus className="w-5 h-5" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <h1 className="text-[15px] font-semibold" style={{ color: "var(--text-primary)" }}>
                {currentChat?.title === "New Chat" ? "Samarth AI" : currentChat?.title || "Samarth AI"}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors md:hidden"
              style={{ color: "var(--text-secondary)" }}
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto scroll-smooth pb-44">
          {currentChat && currentChat.messages.length === 0 ? (
            /* ===================== WELCOME SCREEN ===================== */
            <div className="h-full flex flex-col items-center justify-center p-6 md:p-8 text-center relative overflow-hidden">
              {/* Floating Orbs */}
              <div className="orb orb-1" />
              <div className="orb orb-2" />
              <div className="orb orb-3" />

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="relative z-10 max-w-2xl"
              >
                {/* Logo */}
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.1, duration: 0.5 }}
                  className="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg"
                  style={{ boxShadow: "0 0 40px var(--accent-glow)" }}
                >
                  <Sparkles className="w-8 h-8 text-white" />
                </motion.div>

                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.5 }}
                  className="text-3xl md:text-4xl font-bold mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  How can I help you today?
                </motion.h2>

                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                  className="text-base mb-10"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Agricultural intelligence, data analytics, code generation, and more.
                </motion.p>

                {/* Capability Badges */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.35, duration: 0.5 }}
                  className="flex flex-wrap justify-center gap-2 mb-8"
                >
                  {CAPABILITY_CARDS.map((cap, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
                      style={{
                        background: "var(--bg-tertiary)",
                        color: "var(--text-secondary)",
                        border: "1px solid var(--border-primary)",
                      }}
                    >
                      <cap.icon className="w-3.5 h-3.5" style={{ color: "var(--accent-primary)" }} />
                      {cap.label}
                    </div>
                  ))}
                </motion.div>

                {/* Sample Queries Grid */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4, duration: 0.5 }}
                  className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full"
                >
                  {SAMPLE_QUERIES.map((q, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => sendMessage(q.text)}
                      className="welcome-card flex items-start gap-3 group text-left"
                      style={{ animationDelay: `${i * 0.05}s` }}
                    >
                      <div
                        className={cn(
                          "w-9 h-9 rounded-xl flex items-center justify-center shrink-0 bg-gradient-to-br",
                          q.gradient
                        )}
                      >
                        <q.icon className={cn("w-4.5 h-4.5", q.iconColor)} />
                      </div>
                      <div className="min-w-0">
                        <p
                          className="text-sm font-medium truncate"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {q.text}
                        </p>
                        <p
                          className="text-xs mt-0.5"
                          style={{ color: "var(--text-tertiary)" }}
                        >
                          {q.subtitle}
                        </p>
                      </div>
                    </motion.button>
                  ))}
                </motion.div>

                {/* Developer Credit on Welcome Screen */}
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6, duration: 0.5 }}
                  className="text-[11px] mt-8 font-medium"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Developed by{" "}
                  <span className="gradient-text text-[11px]">Vashista C V</span>
                </motion.p>
              </motion.div>
            </div>
          ) : (
            /* ===================== MESSAGES ===================== */
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-1">
              {currentChat?.messages.map((msg, i) => (
                <ChatMessage
                  key={i}
                  {...msg}
                  isLatest={i === (currentChat?.messages.length ?? 0) - 1 && msg.role === "assistant"}
                  onRegenerate={
                    i === (currentChat?.messages.length ?? 0) - 1 && msg.role === "assistant"
                      ? regenerateResponse
                      : undefined
                  }
                />
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="flex gap-4 w-full message-fade-in">
                  <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center shrink-0 shadow-md">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="pt-2">
                    <div className="typing-indicator">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ===================== INPUT AREA ===================== */}
        <div
          className="w-full p-3 sm:p-4 absolute bottom-0 left-0"
          style={{ background: "var(--bg-primary)" }}
        >
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="relative">
              <div className="input-container flex items-end gap-2 px-4 py-3">
                <textarea
                  ref={textareaRef}
                  className="flex-1 min-h-[24px] max-h-[200px] text-[15px] leading-relaxed py-0"
                  style={{ fontFamily: "var(--font-sans)" }}
                  placeholder="Message Samarth AI..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading}
                  rows={1}
                />
                <div className="flex items-center gap-1 shrink-0 pb-0.5">
                  <VoiceAssistant
                    onTranscript={(t) => setInput(t)}
                    isListening={isListening}
                    setIsListening={setIsListening}
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className={cn("send-btn", input.trim() && !isLoading && "active")}
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </form>
            <p
              className="hidden sm:block text-center text-[11px] mt-2"
              style={{ color: "var(--text-tertiary)" }}
            >
              Samarth AI can make mistakes. Consider checking important information. · Developed by <span className="font-semibold" style={{ color: "var(--text-secondary)" }}>Vashista C V</span>
            </p>
          </div>
        </div>
      </main>

      {/* ===================== ABOUT MODAL ===================== */}
      <AnimatePresence>
        {showAbout && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={() => setShowAbout(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative max-w-md w-full rounded-2xl p-8 shadow-2xl"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-primary)",
              }}
            >
              <button
                onClick={() => setShowAbout(false)}
                className="absolute top-4 right-4 p-1"
                style={{ color: "var(--text-tertiary)" }}
              >
                <X className="w-5 h-5" />
              </button>
              <div className="text-center">
                <div
                  className="w-14 h-14 gradient-bg rounded-2xl flex items-center justify-center mx-auto mb-5 shadow-lg"
                  style={{ boxShadow: "0 0 30px var(--accent-glow)" }}
                >
                  <Sparkles className="w-7 h-7 text-white" />
                </div>
                <h2
                  className="text-xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  Samarth AI
                </h2>
                <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>
                  Advanced Agricultural Intelligence Platform
                </p>
                <div
                  className="rounded-xl p-4 text-left text-sm space-y-2"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-primary)",
                    color: "var(--text-secondary)",
                  }}
                >
                  <div className="flex justify-between">
                    <span>Model</span>
                    <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                      Llama 3.3 70B
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Provider</span>
                    <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                      Groq
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Version</span>
                    <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                      2.0.0
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Backend</span>
                    <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                      FastAPI + LangChain
                    </span>
                  </div>
                </div>
                <p className="text-xs mt-4" style={{ color: "var(--text-tertiary)" }}>
                  Designed & Developed by{" "}
                  <span className="gradient-text">Vashista C V</span>
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ===================== SHORTCUTS MODAL ===================== */}
      <AnimatePresence>
        {showShortcuts && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={() => setShowShortcuts(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative max-w-sm w-full rounded-2xl p-6 shadow-2xl"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-primary)",
              }}
            >
              <button
                onClick={() => setShowShortcuts(false)}
                className="absolute top-4 right-4 p-1"
                style={{ color: "var(--text-tertiary)" }}
              >
                <X className="w-5 h-5" />
              </button>
              <h3
                className="text-lg font-semibold mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Keyboard Shortcuts
              </h3>
              <div className="space-y-3">
                {[
                  { keys: ["Ctrl", "N"], desc: "New chat" },
                  { keys: ["Ctrl", "/"], desc: "Focus input" },
                  { keys: ["Ctrl", "K"], desc: "Search chats" },
                  { keys: ["Enter"], desc: "Send message" },
                  { keys: ["Shift", "Enter"], desc: "New line" },
                  { keys: ["Esc"], desc: "Close modals" },
                ].map((s, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      {s.desc}
                    </span>
                    <div className="flex items-center gap-1">
                      {s.keys.map((k, j) => (
                        <span key={j} className="kbd">
                          {k}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ===================== SIDEBAR CHAT ITEM =====================
function SidebarChatItem({
  chat,
  isActive,
  onSelect,
  onDelete,
  onTogglePin,
  canDelete,
}: {
  chat: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
  onTogglePin: (e: React.MouseEvent) => void;
  canDelete: boolean;
}) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        "sidebar-item group flex items-center gap-3 px-3 py-2.5 text-[13px] cursor-pointer relative",
        isActive ? "active text-white" : "text-gray-400 hover:text-gray-200"
      )}
    >
      <MessageSquare className="w-4 h-4 shrink-0 text-gray-600" />
      <span className="truncate flex-1">{chat.title}</span>

      {/* Actions */}
      <div
        className={cn(
          "flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity",
          isActive && "opacity-100"
        )}
      >
        <button
          onClick={onTogglePin}
          className="p-1 rounded hover:bg-white/10 transition-colors"
          title={chat.pinned ? "Unpin" : "Pin"}
        >
          {chat.pinned ? (
            <PinOff className="w-3 h-3 text-gray-500" />
          ) : (
            <Pin className="w-3 h-3 text-gray-500" />
          )}
        </button>
        {canDelete && (
          <button
            onClick={onDelete}
            className="p-1 rounded hover:bg-white/10 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
}
