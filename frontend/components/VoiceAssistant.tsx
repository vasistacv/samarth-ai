"use client";

import { useState, useEffect, useRef } from "react";
import { Mic, MicOff } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface VoiceAssistantProps {
    onTranscript: (text: string) => void;
    isListening: boolean;
    setIsListening: (listening: boolean) => void;
}

export default function VoiceAssistant({
    onTranscript,
    isListening,
    setIsListening,
}: VoiceAssistantProps) {
    const [isSupported, setIsSupported] = useState(false);
    const recognitionRef = useRef<any>(null);
    const synthRef = useRef<SpeechSynthesis | null>(null);

    useEffect(() => {
        if (typeof window !== "undefined") {
            const SpeechRecognition =
                (window as any).SpeechRecognition ||
                (window as any).webkitSpeechRecognition;

            if (SpeechRecognition && window.speechSynthesis) {
                setIsSupported(true);
                synthRef.current = window.speechSynthesis;

                const recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = "en-US";

                recognition.onresult = (event: any) => {
                    const transcript = event.results[0][0].transcript;
                    onTranscript(transcript);
                    setIsListening(false);
                };

                recognition.onerror = () => {
                    setIsListening(false);
                };

                recognition.onend = () => {
                    setIsListening(false);
                };

                recognitionRef.current = recognition;
            }
        }

        return () => {
            if (recognitionRef.current) recognitionRef.current.stop();
            if (synthRef.current) synthRef.current.cancel();
        };
    }, [onTranscript, setIsListening]);

    const toggleListening = () => {
        if (!isSupported || !recognitionRef.current) return;

        if (isListening) {
            recognitionRef.current.stop();
            setIsListening(false);
        } else {
            recognitionRef.current.start();
            setIsListening(true);
        }
    };

    // Expose speak function
    useEffect(() => {
        (window as any).speakResponse = (text: string) => {
            if (!synthRef.current) return;
            synthRef.current.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            synthRef.current.speak(utterance);
        };
    }, []);

    if (!isSupported) return null;

    return (
        <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={toggleListening}
            className={cn(
                "relative p-2 rounded-full transition-all duration-200",
                isListening
                    ? "text-red-500"
                    : "hover:bg-[var(--bg-tertiary)]"
            )}
            style={{ color: isListening ? undefined : "var(--text-tertiary)" }}
            title={isListening ? "Stop listening" : "Voice input"}
            type="button"
        >
            <AnimatePresence mode="wait">
                {isListening ? (
                    <motion.div
                        key="listening"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="relative"
                    >
                        {/* Ripple rings */}
                        <span className="absolute inset-0 rounded-full bg-red-400/30 animate-ping" />
                        <span
                            className="absolute -inset-1 rounded-full border-2 border-red-400/40"
                            style={{ animation: "ripple 1.5s infinite" }}
                        />
                        <MicOff className="w-5 h-5 relative z-10 text-red-500" />
                    </motion.div>
                ) : (
                    <motion.div
                        key="idle"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                    >
                        <Mic className="w-5 h-5" />
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.button>
    );
}
