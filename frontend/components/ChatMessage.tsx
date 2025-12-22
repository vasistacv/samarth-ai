"use client";

import { motion } from "framer-motion";
import { User, Sparkles, TrendingUp, Copy, Check, RefreshCw, ThumbsUp, ThumbsDown, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ChartRenderer from './ChartRenderer';
import dynamic from 'next/dynamic';
import {
    BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
    AreaChart, Area, ScatterChart, Scatter, RadarChart, Radar,
    PolarGrid, PolarAngleAxis, PolarRadiusAxis, ComposedChart,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// --- Components ---

const CodeBlock = ({ language, children }: any) => {
    const [copied, setCopied] = useState(false);
    const codeText = String(children).replace(/\n$/, "");

    const onCopy = (e: React.MouseEvent) => {
        e.stopPropagation();
        navigator.clipboard.writeText(codeText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="relative group my-4 rounded-xl overflow-hidden border border-gray-700/50 shadow-lg font-mono text-sm">
            {/* Header */}
            <div className="flex items-center justify-between bg-[#1e1e1e] px-4 py-2 text-xs text-gray-400 select-none border-b border-gray-700/50">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50"></span>
                    <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50"></span>
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50"></span>
                    <span className="ml-2 font-mono text-gray-400 uppercase tracking-wider">{language || "Code"}</span>
                </div>
                <button
                    onClick={onCopy}
                    className="flex items-center gap-1.5 hover:text-white transition-colors py-1 px-2 rounded hover:bg-gray-700/50"
                >
                    {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span className="text-[10px] uppercase font-semibold">{copied ? "Copied" : "Copy"}</span>
                </button>
            </div>
            {/* Syntax Highlighter */}
            <SyntaxHighlighter
                language={language}
                style={vscDarkPlus}
                customStyle={{ margin: 0, padding: '1.25rem', background: '#0d0d0d', fontSize: '0.85rem', lineHeight: '1.6' }}
                wrapLines={true}
                showLineNumbers={true}
                lineNumberStyle={{ minWidth: "2.5em", paddingRight: "1em", color: "#555", textAlign: "right" }}
            >
                {codeText}
            </SyntaxHighlighter>
        </div>
    );
};

// Ultra-simple crash-proof chart component
const InlineChart = ({ chartJson }: { chartJson: string }) => {
    const [mounted, setMounted] = useState(false);
    const [error, setError] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return <div className="w-full my-4 p-4 bg-gray-100 rounded-xl animate-pulse"><div className="h-64 bg-gray-200 rounded"></div></div>;
    }

    if (error) {
        return null;
    }

    try {
        const config = JSON.parse(chartJson);
        if (!config || !config.data || !Array.isArray(config.data) || config.data.length === 0) {
            return null;
        }

        const { type, title, data, xKey } = config;
        const keys = Object.keys(data[0] || {});
        const labelKey = xKey || keys[0] || 'name';
        const valueKeys = keys.filter(k => k !== labelKey);

        if (valueKeys.length === 0) return null;

        const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

        // Simple bar chart only for stability
        return (
            <div className="w-full my-4 p-4 bg-white rounded-xl border border-gray-200 shadow">
                <div className="text-sm font-semibold text-gray-700 mb-3">📊 {title || 'Chart'}</div>
                <div style={{ width: '100%', height: 280 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey={labelKey} tick={{ fontSize: 10 }} />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip />
                            <Legend />
                            {valueKeys.slice(0, 5).map((key, i) => (
                                <Bar key={key} dataKey={key} fill={COLORS[i % COLORS.length]} />
                            ))}
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    } catch (e) {
        console.error('Chart error:', e);
        return null;
    }
};

interface ChatMessageProps {
    role: "user" | "assistant";
    content: string;
    timestamp: string;
    structuredData?: {
        headers: string[];
        rows: any[][];
        sql?: string;
    };
    isLatest?: boolean;
    onRegenerate?: () => void;
}

export default function ChatMessage({
    role,
    content,
    timestamp,
    structuredData,
    isLatest,
    onRegenerate
}: ChatMessageProps) {
    const isUser = role === "user";
    const [copied, setCopied] = useState(false);

    // Typewriter State
    const [displayedContent, setDisplayedContent] = useState(isUser ? content : "");
    const [isTyping, setIsTyping] = useState(!isUser && isLatest);
    const indexRef = useRef(0);

    // Effect for Typewriter
    useEffect(() => {
        if (isUser) {
            setDisplayedContent(content);
            return;
        }

        if (!isLatest) {
            // If not latest, show full content immediately (history)
            setDisplayedContent(content);
            setIsTyping(false);
            return;
        }

        // If latest and assistant, animate
        setIsTyping(true);
        setDisplayedContent("");
        indexRef.current = 0;

        const intervalId = setInterval(() => {
            if (indexRef.current >= content.length) {
                clearInterval(intervalId);
                setIsTyping(false);
                setDisplayedContent(content);
                return;
            }

            // Smoother typing: Variable speed based on punctuation
            const char = content[indexRef.current];
            let increment = 1;
            let delay = 15; // Standard typing speed

            // Speed up for long code blocks or spaces
            if (content.length > 500) {
                increment = 2;
                delay = 5;
            }

            // Render chunk
            const nextIndex = Math.min(indexRef.current + increment, content.length);
            setDisplayedContent(content.substring(0, nextIndex));
            indexRef.current = nextIndex;

        }, 10); // Check loop

        return () => clearInterval(intervalId);
    }, [content, isLatest, isUser]);

    const handleCopy = () => {
        navigator.clipboard.writeText(content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={cn("flex gap-4 w-full group mb-6", isUser ? "justify-end" : "justify-start")}>

            {!isUser && (
                <div className="flex-shrink-0 w-8 h-8 mt-1 rounded-full border border-gray-200 bg-white flex items-center justify-center shadow-sm">
                    <Sparkles className="w-4 h-4 text-blue-600" />
                </div>
            )}

            <div className={cn("flex flex-col max-w-[90%] lg:max-w-[85%]", isUser && "items-end")}>
                <span className="text-[12px] text-gray-400 mb-1 px-1 font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    {isUser ? "You" : "Samarth AI"}
                </span>

                <div
                    className={cn(
                        "relative px-4 py-3 text-[15px] leading-7",
                        isUser
                            ? "bg-[#f4f4f4] text-gray-900 rounded-[24px] rounded-tr-md px-6 py-3.5" // ChatGPT User Style
                            : "bg-transparent text-gray-800 p-0"
                    )}
                >
                    {!content.includes("[[STRUCTURED_RESULT::") && (
                        <div className={cn(
                            "prose prose-neutral max-w-none break-words",
                            isUser ? "prose-p:text-gray-900" : "prose-p:text-gray-800",
                            "prose-p:my-1.5 prose-p:leading-relaxed", // Relaxed leading for smoothness
                            "prose-headings:font-semibold prose-headings:text-gray-900 prose-headings:mt-6 prose-headings:mb-3",
                            "prose-li:my-0.5",
                            "prose-pre:m-0 prose-pre:p-0 prose-pre:bg-transparent"
                        )}
                        >
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    code({ node, className, children, ...props }: any) {
                                        const match = /language-(\S+)/.exec(className || "");
                                        const lang = match ? match[1] : null;
                                        const codeText = String(children).replace(/\n$/, "");

                                        // Check for json:chart special block OR plain json with chart structure
                                        if (lang === 'json:chart' || lang === 'chart') {
                                            return <InlineChart chartJson={codeText} />;
                                        }

                                        // Also detect plain 'json' blocks that contain chart data
                                        if (lang === 'json') {
                                            try {
                                                const parsed = JSON.parse(codeText);
                                                // If it has chart properties, render as chart
                                                if (parsed.type && parsed.data && Array.isArray(parsed.data)) {
                                                    return <InlineChart chartJson={codeText} />;
                                                }
                                            } catch (e) {
                                                // Not valid JSON or not a chart, render as code
                                            }
                                        }

                                        // Regular code block
                                        if (match) {
                                            return (
                                                <CodeBlock language={lang}>
                                                    {children}
                                                </CodeBlock>
                                            );
                                        }

                                        // Inline code
                                        return (
                                            <code className={cn(
                                                "px-1.5 py-0.5 rounded text-[13px] font-mono",
                                                isUser ? "bg-white/50 text-gray-800" : "bg-gray-100 text-gray-800 border border-gray-200"
                                            )} {...props}>
                                                {children}
                                            </code>
                                        );
                                    },
                                    pre: ({ children }: any) => <>{children}</>,
                                    a: ({ href, children }: any) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{children}</a>,
                                    // Table Components
                                    table: ({ children }: any) => (
                                        <div className="overflow-x-auto my-4 rounded-lg border border-gray-200 shadow-sm">
                                            <table className="min-w-full divide-y divide-gray-200 text-sm">
                                                {children}
                                            </table>
                                        </div>
                                    ),
                                    thead: ({ children }: any) => <thead className="bg-gray-50">{children}</thead>,
                                    tbody: ({ children }: any) => <tbody className="bg-white divide-y divide-gray-100">{children}</tbody>,
                                    tr: ({ children }: any) => <tr className="hover:bg-gray-50 transition-colors">{children}</tr>,
                                    th: ({ children }: any) => <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wide">{children}</th>,
                                    td: ({ children }: any) => <td className="px-4 py-3 text-gray-700 whitespace-nowrap">{children}</td>,
                                }}
                            >
                                {displayedContent}
                            </ReactMarkdown>

                            {/* Blinking Cursor for AI */}
                            {!isUser && isTyping && (
                                <span className="inline-block w-2 h-4 bg-blue-500 ml-1 rounded-sm animate-pulse align-middle"></span>
                            )}
                        </div>
                    )}

                    {/* Footer Actions (AI Only) */}
                    {!isUser && !isTyping && (
                        <div className={cn(
                            "flex items-center gap-4 mt-3 text-gray-400 transition-opacity duration-200",
                            isLatest ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                        )}>
                            {/* Copy */}
                            <button
                                onClick={handleCopy}
                                className="flex items-center gap-1 hover:text-gray-600 transition-colors tooltip"
                                title="Copy message"
                            >
                                {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                            </button>

                            {/* Regenerate (Only valid if it's the latest message) */}
                            {isLatest && onRegenerate && (
                                <button
                                    onClick={onRegenerate}
                                    className="flex items-center gap-1 hover:text-gray-600 transition-colors"
                                    title="Regenerate response"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                </button>
                            )}

                            {/* Feedback (Visual Only for now) */}
                            <button className="hover:text-gray-600 transition-colors"><ThumbsUp className="w-4 h-4" /></button>
                            <button className="hover:text-gray-600 transition-colors"><ThumbsDown className="w-4 h-4" /></button>
                        </div>
                    )}

                    {/* Structured Data Table + Chart */}
                    {structuredData && structuredData.headers && structuredData.rows && structuredData.rows.length > 0 && (
                        <div className="mt-5 mb-2 animate-in fade-in zoom-in duration-300 space-y-4">
                            {/* Chart Visualization */}
                            <ChartRenderer data={{ headers: structuredData.headers, rows: structuredData.rows }} />

                            {/* Data Table with Download */}
                            <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white shadow-sm">
                                <div className="flex items-center justify-between px-4 py-2 bg-gray-50/50 border-b border-gray-100">
                                    <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Data Table</span>
                                    <button
                                        onClick={() => {
                                            const csv = [
                                                structuredData.headers.join(','),
                                                ...structuredData.rows.map((row: any[]) => row.join(','))
                                            ].join('\n');
                                            const blob = new Blob([csv], { type: 'text/csv' });
                                            const url = URL.createObjectURL(blob);
                                            const a = document.createElement('a');
                                            a.href = url;
                                            a.download = 'data.csv';
                                            a.click();
                                        }}
                                        className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-500 hover:text-green-600 hover:bg-green-50 rounded-md transition-colors"
                                        title="Download as CSV"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        <span className="hidden sm:inline">CSV</span>
                                    </button>
                                </div>
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50/30">
                                        <tr>
                                            {structuredData.headers.map((header: string, i: number) => (
                                                <th key={i} className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wide">
                                                    {header.replace(/_/g, " ")}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-100">
                                        {structuredData.rows.map((row: any[], i: number) => (
                                            <tr key={i} className="hover:bg-gray-50 transition-colors">
                                                {row.map((cell: any, j: number) => (
                                                    <td key={j} className="px-4 py-3 text-gray-700 whitespace-nowrap font-medium">{cell}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
