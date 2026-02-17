"use client";

import { motion } from "framer-motion";
import {
    User,
    Sparkles,
    Copy,
    Check,
    RefreshCw,
    ThumbsUp,
    ThumbsDown,
    Download,
    Share2,
    Volume2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import ChartRenderer from "./ChartRenderer";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

// --- Code Block Component ---
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
        <div className="relative group my-4 rounded-xl overflow-hidden shadow-lg font-mono text-sm" style={{ border: "1px solid #2d2d2d" }}>
            <div className="code-block-header">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#ff5f56" }} />
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#ffbd2e" }} />
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#27c93f" }} />
                    <span className="ml-2 font-mono text-gray-500 uppercase tracking-wider text-[10px]">
                        {language || "code"}
                    </span>
                </div>
                <button
                    onClick={onCopy}
                    className="flex items-center gap-1.5 hover:text-white transition-colors py-1 px-2 rounded hover:bg-gray-700/50"
                >
                    {copied ? (
                        <Check className="w-3.5 h-3.5 text-green-400" />
                    ) : (
                        <Copy className="w-3.5 h-3.5" />
                    )}
                    <span className="text-[10px] uppercase font-semibold">
                        {copied ? "Copied!" : "Copy"}
                    </span>
                </button>
            </div>
            <SyntaxHighlighter
                language={language}
                style={vscDarkPlus}
                customStyle={{
                    margin: 0,
                    padding: "1.25rem",
                    background: "#0d0d0d",
                    fontSize: "0.85rem",
                    lineHeight: "1.7",
                }}
                wrapLines={true}
                showLineNumbers={true}
                lineNumberStyle={{
                    minWidth: "2.5em",
                    paddingRight: "1em",
                    color: "#444",
                    textAlign: "right",
                }}
            >
                {codeText}
            </SyntaxHighlighter>
        </div>
    );
};

// --- Inline Chart Component ---
const InlineChart = ({ chartJson }: { chartJson: string }) => {
    const [mounted, setMounted] = useState(false);

    useEffect(() => setMounted(true), []);

    if (!mounted) {
        return (
            <div className="w-full my-4 p-4 rounded-xl shimmer" style={{ height: 280 }} />
        );
    }

    try {
        const config = JSON.parse(chartJson);
        if (!config?.data || !Array.isArray(config.data) || config.data.length === 0) return null;

        const { title, data, xKey } = config;
        const keys = Object.keys(data[0] || {});
        const labelKey = xKey || keys[0] || "name";
        const valueKeys = keys.filter((k) => k !== labelKey);

        if (valueKeys.length === 0) return null;

        const COLORS = ["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#a855f7"];

        return (
            <div className="chart-container my-4">
                <div
                    className="text-xs font-semibold uppercase tracking-wider mb-4 flex items-center gap-2"
                    style={{ color: "var(--text-secondary)" }}
                >
                    <BarChart className="w-3.5 h-3.5" style={{ color: "var(--accent-primary)" }} />
                    {title || "Chart"}
                </div>
                <div style={{ width: "100%", height: 280 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                            <XAxis dataKey={labelKey} tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} />
                            <YAxis tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} />
                            <Tooltip
                                contentStyle={{
                                    background: "var(--bg-primary)",
                                    border: "1px solid var(--border-primary)",
                                    borderRadius: "8px",
                                    boxShadow: "var(--shadow-md)",
                                    color: "var(--text-primary)",
                                    fontSize: 12,
                                }}
                            />
                            <Legend />
                            {valueKeys.slice(0, 5).map((key, i) => (
                                <Bar
                                    key={key}
                                    dataKey={key}
                                    fill={COLORS[i % COLORS.length]}
                                    radius={[4, 4, 0, 0]}
                                />
                            ))}
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    } catch {
        return null;
    }
};

// --- Main Component ---
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
    onRegenerate,
}: ChatMessageProps) {
    const isUser = role === "user";
    const [copied, setCopied] = useState(false);
    const [liked, setLiked] = useState<"up" | "down" | null>(null);

    // Typewriter State
    const [displayedContent, setDisplayedContent] = useState(isUser ? content : "");
    const [isTyping, setIsTyping] = useState(!isUser && isLatest);
    const indexRef = useRef(0);

    useEffect(() => {
        if (isUser) {
            setDisplayedContent(content);
            return;
        }

        if (!isLatest) {
            setDisplayedContent(content);
            setIsTyping(false);
            return;
        }

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

            let increment = content.length > 500 ? 3 : content.length > 200 ? 2 : 1;
            const nextIndex = Math.min(indexRef.current + increment, content.length);
            setDisplayedContent(content.substring(0, nextIndex));
            indexRef.current = nextIndex;
        }, 8);

        return () => clearInterval(intervalId);
    }, [content, isLatest, isUser]);

    const handleCopy = () => {
        navigator.clipboard.writeText(content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleSpeak = () => {
        if (typeof window !== "undefined" && window.speechSynthesis) {
            window.speechSynthesis.cancel();
            const clean = content.replace(/[#*`_~\[\]()]/g, "").slice(0, 500);
            const utterance = new SpeechSynthesisUtterance(clean);
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    };

    return (
        <div className={cn("flex gap-3 w-full group mb-2 message-fade-in", isUser ? "justify-end" : "justify-start")}>
            {/* AI Avatar */}
            {!isUser && (
                <div className="flex-shrink-0 w-8 h-8 mt-1 rounded-full gradient-bg flex items-center justify-center shadow-md">
                    <Sparkles className="w-4 h-4 text-white" />
                </div>
            )}

            <div className={cn("flex flex-col max-w-[88%] lg:max-w-[82%]", isUser && "items-end")}>
                {/* Message Bubble */}
                <div
                    className={cn(
                        "relative text-[15px] leading-7",
                        isUser
                            ? "msg-user px-5 py-3.5"
                            : "msg-ai pt-1"
                    )}
                >
                    {!content.includes("[[STRUCTURED_RESULT::") && (
                        <div
                            className={cn(
                                "prose prose-neutral max-w-none break-words prose-themed",
                                "prose-p:my-1.5 prose-p:leading-relaxed",
                                "prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3",
                                "prose-li:my-0.5",
                                "prose-pre:m-0 prose-pre:p-0 prose-pre:bg-transparent",
                                "prose-strong:font-semibold",
                            )}
                            style={{ color: "var(--text-primary)" }}
                        >
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    code({ node, className, children, ...props }: any) {
                                        const match = /language-(\S+)/.exec(className || "");
                                        const lang = match ? match[1] : null;
                                        const codeText = String(children).replace(/\n$/, "");

                                        if (lang === "json:chart" || lang === "chart") {
                                            return <InlineChart chartJson={codeText} />;
                                        }

                                        if (lang === "json") {
                                            try {
                                                const parsed = JSON.parse(codeText);
                                                if (parsed.type && parsed.data && Array.isArray(parsed.data)) {
                                                    return <InlineChart chartJson={codeText} />;
                                                }
                                            } catch { }
                                        }

                                        if (match) {
                                            return <CodeBlock language={lang}>{children}</CodeBlock>;
                                        }

                                        return (
                                            <code
                                                className="px-1.5 py-0.5 rounded text-[13px] font-mono"
                                                style={{
                                                    background: "var(--bg-tertiary)",
                                                    color: "var(--accent-primary)",
                                                    border: "1px solid var(--border-primary)",
                                                }}
                                                {...props}
                                            >
                                                {children}
                                            </code>
                                        );
                                    },
                                    pre: ({ children }: any) => <>{children}</>,
                                    a: ({ href, children }: any) => (
                                        <a
                                            href={href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            style={{ color: "var(--accent-primary)" }}
                                            className="hover:underline"
                                        >
                                            {children}
                                        </a>
                                    ),
                                    table: ({ children }: any) => (
                                        <div className="overflow-x-auto my-4 data-table">
                                            <table className="min-w-full divide-y text-sm" style={{ borderColor: "var(--border-primary)" }}>
                                                {children}
                                            </table>
                                        </div>
                                    ),
                                    thead: ({ children }: any) => (
                                        <thead style={{ background: "var(--bg-tertiary)" }}>{children}</thead>
                                    ),
                                    tbody: ({ children }: any) => (
                                        <tbody className="divide-y" style={{ borderColor: "var(--border-secondary)" }}>
                                            {children}
                                        </tbody>
                                    ),
                                    tr: ({ children }: any) => (
                                        <tr className="transition-colors" style={{ borderColor: "var(--border-secondary)" }}>
                                            {children}
                                        </tr>
                                    ),
                                    th: ({ children }: any) => (
                                        <th
                                            className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wider"
                                            style={{ color: "var(--text-secondary)" }}
                                        >
                                            {children}
                                        </th>
                                    ),
                                    td: ({ children }: any) => (
                                        <td
                                            className="px-4 py-3 text-[13px] whitespace-nowrap"
                                            style={{ color: "var(--text-primary)" }}
                                        >
                                            {children}
                                        </td>
                                    ),
                                }}
                            >
                                {displayedContent}
                            </ReactMarkdown>

                            {/* Typing cursor */}
                            {!isUser && isTyping && (
                                <span
                                    className="inline-block w-0.5 h-5 ml-0.5 rounded-sm animate-pulse align-middle"
                                    style={{ background: "var(--accent-primary)" }}
                                />
                            )}
                        </div>
                    )}

                    {/* Action Bar (AI Only) */}
                    {!isUser && !isTyping && (
                        <div
                            className={cn(
                                "flex items-center gap-1 mt-3 transition-opacity duration-200",
                                isLatest ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                            )}
                        >
                            {/* Copy */}
                            <button
                                onClick={handleCopy}
                                className="action-btn tooltip-wrapper"
                                data-tooltip="Copy"
                            >
                                {copied ? <Check className="w-4 h-4" style={{ color: "#34a853" }} /> : <Copy className="w-4 h-4" />}
                            </button>

                            {/* Regenerate */}
                            {isLatest && onRegenerate && (
                                <button
                                    onClick={onRegenerate}
                                    className="action-btn tooltip-wrapper"
                                    data-tooltip="Regenerate"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                </button>
                            )}

                            {/* Read Aloud */}
                            <button
                                onClick={handleSpeak}
                                className="action-btn tooltip-wrapper"
                                data-tooltip="Read aloud"
                            >
                                <Volume2 className="w-4 h-4" />
                            </button>

                            {/* Divider */}
                            <div className="w-px h-4 mx-1" style={{ background: "var(--border-primary)" }} />

                            {/* Like/Dislike */}
                            <button
                                onClick={() => setLiked(liked === "up" ? null : "up")}
                                className={cn("action-btn", liked === "up" && "liked")}
                            >
                                <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setLiked(liked === "down" ? null : "down")}
                                className={cn("action-btn", liked === "down" && "liked")}
                            >
                                <ThumbsDown className="w-4 h-4" />
                            </button>
                        </div>
                    )}

                    {/* Structured Data Table + Chart */}
                    {structuredData &&
                        structuredData.headers &&
                        structuredData.rows &&
                        structuredData.rows.length > 0 && (
                            <div className="mt-5 mb-2 space-y-4 message-fade-in">
                                {/* Chart */}
                                <ChartRenderer
                                    data={{
                                        headers: structuredData.headers,
                                        rows: structuredData.rows,
                                    }}
                                />

                                {/* Data Table */}
                                <div className="data-table overflow-x-auto">
                                    <div
                                        className="flex items-center justify-between px-4 py-2.5"
                                        style={{
                                            background: "var(--bg-tertiary)",
                                            borderBottom: "1px solid var(--border-primary)",
                                        }}
                                    >
                                        <span
                                            className="text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1.5"
                                            style={{ color: "var(--text-secondary)" }}
                                        >
                                            📊 Data Results
                                        </span>
                                        <button
                                            onClick={() => {
                                                const csv = [
                                                    structuredData.headers.join(","),
                                                    ...structuredData.rows.map((row: any[]) => row.join(",")),
                                                ].join("\n");
                                                const blob = new Blob([csv], { type: "text/csv" });
                                                const url = URL.createObjectURL(blob);
                                                const a = document.createElement("a");
                                                a.href = url;
                                                a.download = "samarth_data.csv";
                                                a.click();
                                            }}
                                            className="action-btn flex items-center gap-1 text-[11px] font-medium"
                                        >
                                            <Download className="w-3.5 h-3.5" />
                                            <span className="hidden sm:inline">Export CSV</span>
                                        </button>
                                    </div>
                                    <table className="min-w-full text-sm">
                                        <thead>
                                            <tr>
                                                {structuredData.headers.map((header: string, i: number) => (
                                                    <th
                                                        key={i}
                                                        className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wider"
                                                        style={{
                                                            background: "var(--bg-tertiary)",
                                                            color: "var(--text-secondary)",
                                                        }}
                                                    >
                                                        {header.replace(/_/g, " ")}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {structuredData.rows.map((row: any[], i: number) => (
                                                <tr
                                                    key={i}
                                                    className="transition-colors hover:bg-[var(--bg-secondary)]"
                                                    style={{
                                                        borderTop: "1px solid var(--border-secondary)",
                                                    }}
                                                >
                                                    {row.map((cell: any, j: number) => (
                                                        <td
                                                            key={j}
                                                            className="px-4 py-3 text-[13px] whitespace-nowrap font-medium"
                                                            style={{ color: "var(--text-primary)" }}
                                                        >
                                                            {cell}
                                                        </td>
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

            {/* User Avatar */}
            {isUser && (
                <div
                    className="flex-shrink-0 w-8 h-8 mt-1 rounded-full flex items-center justify-center"
                    style={{
                        background: "var(--accent-primary)",
                    }}
                >
                    <User className="w-4 h-4 text-white" />
                </div>
            )}
        </div>
    );
}
