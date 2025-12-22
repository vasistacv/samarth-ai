"use client";

import { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Volume2, VolumeX } from "lucide-react";
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
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isSupported, setIsSupported] = useState(false);
    const recognitionRef = useRef<any>(null);
    const synthRef = useRef<SpeechSynthesis | null>(null);

    useEffect(() => {
        // Check for browser support
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

                recognition.onerror = (event: any) => {
                    console.error("Speech recognition error:", event.error);
                    setIsListening(false);
                };

                recognition.onend = () => {
                    setIsListening(false);
                };

                recognitionRef.current = recognition;
            }
        }

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            if (synthRef.current) {
                synthRef.current.cancel();
            }
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

    const speak = (text: string) => {
        if (!synthRef.current) return;

        synthRef.current.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);

        synthRef.current.speak(utterance);
    };

    const stopSpeaking = () => {
        if (synthRef.current) {
            synthRef.current.cancel();
            setIsSpeaking(false);
        }
    };

    // Expose speak function to parent
    useEffect(() => {
        (window as any).speakResponse = speak;
    }, []);

    if (!isSupported) {
        return null;
    }

    return (
        <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleListening}
            className={cn(
                "relative p-2 rounded-full transition-colors duration-200",
                isListening
                    ? "bg-red-50 text-red-500"
                    : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
            )}
            title={isListening ? "Stop listening" : "Voice input"}
            type="button"
        >
            <AnimatePresence mode="wait">
                {isListening ? (
                    <motion.div
                        key="mic-off"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                    >
                        <div className="absolute inset-0 rounded-full animate-ping bg-red-200 opacity-75"></div>
                        <MicOff className="w-5 h-5 relative z-10" />
                    </motion.div>
                ) : (
                    <motion.div
                        key="mic-on"
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
