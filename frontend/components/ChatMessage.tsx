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
    Volume2,
    BarChart3,
    TrendingUp,
    Target,
    Activity,
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
    LineChart,
    Line,
    AreaChart,
    Area,
    PieChart,
    Pie,
    Cell,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ScatterChart,
    Scatter,
    ComposedChart,
    XAxis,
    YAxis,
    ZAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";

// ============================================================
// CONSTANTS
// ============================================================
const CHART_COLORS = [
    "#4285f4", "#34a853", "#fbbc05", "#ea4335", "#a855f7",
    "#06b6d4", "#f97316", "#ec4899", "#14b8a6", "#8b5cf6",
];

const CHART_GRADIENTS = [
    { id: "icg1", start: "#4285f4", end: "#1a73e8" },
    { id: "icg2", start: "#34a853", end: "#1e8e3e" },
    { id: "icg3", start: "#fbbc05", end: "#f9ab00" },
    { id: "icg4", start: "#ea4335", end: "#d93025" },
    { id: "icg5", start: "#a855f7", end: "#7c3aed" },
    { id: "icg6", start: "#06b6d4", end: "#0891b2" },
    { id: "icg7", start: "#f97316", end: "#ea580c" },
    { id: "icg8", start: "#ec4899", end: "#db2777" },
];

const chartTooltipStyle = {
    contentStyle: {
        background: "var(--bg-primary)",
        border: "1px solid var(--border-primary)",
        borderRadius: "10px",
        boxShadow: "var(--shadow-md)",
        color: "var(--text-primary)",
        fontSize: 12,
        padding: "10px 14px",
    },
    labelStyle: {
        color: "var(--text-secondary)",
        fontWeight: 600,
        marginBottom: 4,
    },
};

// ============================================================
// CODE BLOCK COMPONENT
// ============================================================
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

// ============================================================
// ADVANCED INLINE CHART COMPONENT
// ============================================================
const InlineChart = ({ chartJson }: { chartJson: string }) => {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    if (!mounted) {
        return <div className="w-full my-4 p-4 rounded-xl shimmer" style={{ height: 320 }} />;
    }

    try {
        const config = JSON.parse(chartJson);
        if (!config?.data || !Array.isArray(config.data) || config.data.length === 0) return null;

        const { type = "bar", title, data, xKey } = config;
        const keys = Object.keys(data[0] || {});
        const labelKey = xKey || keys[0] || "name";
        const valueKeys = keys.filter((k) => k !== labelKey);

        if (valueKeys.length === 0) return null;

        const chartType = type.toLowerCase();

        const ChartIcon =
            chartType === "line" || chartType === "area" ? TrendingUp
                : chartType === "pie" ? BarChart3
                    : chartType === "radar" ? Target
                        : chartType === "scatter" ? Activity
                            : BarChart3;

        return (
            <div className="chart-container my-5">
                {/* Chart Header */}
                <div
                    className="flex items-center justify-between mb-4 pb-2.5"
                    style={{ borderBottom: "1px solid var(--border-secondary)" }}
                >
                    <div className="flex items-center gap-2">
                        <div
                            className="w-7 h-7 rounded-lg flex items-center justify-center"
                            style={{ background: "rgba(66, 133, 244, 0.12)" }}
                        >
                            <ChartIcon className="w-3.5 h-3.5" style={{ color: "var(--accent-primary)" }} />
                        </div>
                        <span className="text-xs font-bold" style={{ color: "var(--text-primary)" }}>
                            {title || "Chart"}
                        </span>
                    </div>
                    <span
                        className="text-[10px] font-semibold px-2 py-1 rounded-full uppercase tracking-wider"
                        style={{
                            background: "var(--bg-tertiary)",
                            color: "var(--text-tertiary)",
                            border: "1px solid var(--border-primary)",
                        }}
                    >
                        {chartType} • {data.length} pts
                    </span>
                </div>

                {/* Chart Body */}
                <div style={{ width: "100%", height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        {/* BAR CHART */}
                        {chartType === "bar" ? (
                            <BarChart data={data}>
                                <defs>
                                    {CHART_GRADIENTS.map((g) => (
                                        <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor={g.start} stopOpacity={0.9} />
                                            <stop offset="100%" stopColor={g.end} stopOpacity={0.5} />
                                        </linearGradient>
                                    ))}
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {valueKeys.slice(0, 6).map((key, i) => (
                                    <Bar key={key} dataKey={key} fill={`url(#${CHART_GRADIENTS[i % CHART_GRADIENTS.length].id})`} radius={[6, 6, 0, 0]} />
                                ))}
                            </BarChart>
                        ) : chartType === "line" ? (
                            /* LINE CHART */
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {valueKeys.slice(0, 6).map((key, i) => (
                                    <Line
                                        key={key} type="monotone" dataKey={key}
                                        stroke={CHART_COLORS[i % CHART_COLORS.length]} strokeWidth={2.5}
                                        dot={{ fill: CHART_COLORS[i % CHART_COLORS.length], r: 4, stroke: "var(--bg-primary)", strokeWidth: 2 }}
                                        activeDot={{ r: 6 }}
                                    />
                                ))}
                            </LineChart>
                        ) : chartType === "area" ? (
                            /* AREA CHART */
                            <AreaChart data={data}>
                                <defs>
                                    {CHART_GRADIENTS.map((g) => (
                                        <linearGradient key={`a-${g.id}`} id={`a-${g.id}`} x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor={g.start} stopOpacity={0.35} />
                                            <stop offset="100%" stopColor={g.end} stopOpacity={0.05} />
                                        </linearGradient>
                                    ))}
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {valueKeys.slice(0, 6).map((key, i) => (
                                    <Area key={key} type="monotone" dataKey={key} stroke={CHART_GRADIENTS[i % CHART_GRADIENTS.length].start} fill={`url(#a-${CHART_GRADIENTS[i % CHART_GRADIENTS.length].id})`} strokeWidth={2.5} />
                                ))}
                            </AreaChart>
                        ) : chartType === "pie" ? (
                            /* PIE / DONUT CHART */
                            <PieChart>
                                <defs>
                                    {CHART_COLORS.map((c, i) => (
                                        <linearGradient key={`pie-${i}`} id={`ipie-${i}`} x1="0" y1="0" x2="1" y2="1">
                                            <stop offset="0%" stopColor={c} stopOpacity={0.95} />
                                            <stop offset="100%" stopColor={c} stopOpacity={0.6} />
                                        </linearGradient>
                                    ))}
                                </defs>
                                <Pie
                                    data={data}
                                    dataKey={valueKeys[0]}
                                    nameKey={labelKey}
                                    cx="50%" cy="50%"
                                    outerRadius={105} innerRadius={45}
                                    paddingAngle={3}
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    labelLine={{ stroke: "var(--text-tertiary)" }}
                                >
                                    {data.map((_: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={`url(#ipie-${index % CHART_COLORS.length})`} stroke="var(--bg-primary)" strokeWidth={2} />
                                    ))}
                                </Pie>
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                            </PieChart>
                        ) : chartType === "radar" ? (
                            /* RADAR CHART */
                            <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
                                <PolarGrid stroke="var(--border-primary)" />
                                <PolarAngleAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} />
                                <PolarRadiusAxis tick={{ fontSize: 9, fill: "var(--text-tertiary)" }} />
                                {valueKeys.slice(0, 4).map((key, i) => (
                                    <Radar key={key} name={key} dataKey={key} stroke={CHART_COLORS[i % CHART_COLORS.length]} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.15} strokeWidth={2} />
                                ))}
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                            </RadarChart>
                        ) : chartType === "scatter" ? (
                            /* SCATTER CHART */
                            <ScatterChart>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={valueKeys[0] || labelKey} name={valueKeys[0]} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <YAxis dataKey={valueKeys[1] || valueKeys[0]} name={valueKeys[1] || valueKeys[0]} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <ZAxis range={[60, 300]} />
                                <Tooltip {...chartTooltipStyle} />
                                <Scatter name="Data" data={data} fill={CHART_COLORS[0]} fillOpacity={0.7} />
                            </ScatterChart>
                        ) : chartType === "composed" ? (
                            /* COMPOSED (BAR + LINE) CHART */
                            <ComposedChart data={data}>
                                <defs>
                                    {CHART_GRADIENTS.map((g) => (
                                        <linearGradient key={`c-${g.id}`} id={`c-${g.id}`} x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor={g.start} stopOpacity={0.8} />
                                            <stop offset="100%" stopColor={g.end} stopOpacity={0.4} />
                                        </linearGradient>
                                    ))}
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                {valueKeys.length > 1 && (
                                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                                )}
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {valueKeys.map((key, idx) =>
                                    idx === 0 ? (
                                        <Bar key={key} yAxisId="left" dataKey={key} fill={`url(#c-${CHART_GRADIENTS[idx % CHART_GRADIENTS.length].id})`} radius={[6, 6, 0, 0]} />
                                    ) : (
                                        <Line key={key} yAxisId={idx === 1 && valueKeys.length > 1 ? "right" : "left"} type="monotone" dataKey={key} stroke={CHART_COLORS[idx % CHART_COLORS.length]} strokeWidth={2.5} dot={{ fill: CHART_COLORS[idx % CHART_COLORS.length], r: 4, stroke: "var(--bg-primary)", strokeWidth: 2 }} />
                                    )
                                )}
                            </ComposedChart>
                        ) : chartType === "gauge" ? (
                            /* GAUGE / KPI CHART */
                            <PieChart>
                                <Pie
                                    data={[
                                        { name: valueKeys[0], value: Number(data[0]?.[valueKeys[0]] || 0) },
                                        { name: "remaining", value: Math.max(0, 100 - Number(data[0]?.[valueKeys[0]] || 0)) },
                                    ]}
                                    dataKey="value"
                                    startAngle={200}
                                    endAngle={-20}
                                    cx="50%" cy="55%"
                                    innerRadius={70} outerRadius={100}
                                    paddingAngle={0}
                                >
                                    <Cell fill={CHART_COLORS[0]} />
                                    <Cell fill="var(--border-primary)" />
                                </Pie>
                                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 28, fontWeight: 700, fill: "var(--text-primary)" }}>
                                    {data[0]?.[valueKeys[0]] || 0}%
                                </text>
                                <text x="50%" y="62%" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 11, fill: "var(--text-tertiary)" }}>
                                    {valueKeys[0]?.replace(/_/g, " ")}
                                </text>
                            </PieChart>
                        ) : (
                            /* FALLBACK: BAR */
                            <BarChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis dataKey={labelKey} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} />
                                <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} />
                                <Tooltip {...chartTooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 11 }} />
                                {valueKeys.slice(0, 6).map((key, i) => (
                                    <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
                                ))}
                            </BarChart>
                        )}
                    </ResponsiveContainer>
                </div>
            </div>
        );
    } catch {
        return null;
    }
};

// ============================================================
// MAIN COMPONENT
// ============================================================
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

            let increment = content.length > 500 ? 4 : content.length > 200 ? 2 : 1;
            const nextIndex = Math.min(indexRef.current + increment, content.length);
            setDisplayedContent(content.substring(0, nextIndex));
            indexRef.current = nextIndex;
        }, 6);

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
                        "relative text-[15px] leading-relaxed",
                        isUser ? "msg-user px-5 py-3.5" : "msg-ai py-1 px-1"
                    )}
                >
                    {!content.includes("[[STRUCTURED_RESULT::") && (
                        <div
                            className={cn(
                                "prose prose-neutral max-w-none break-words prose-themed"
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

                                        // Handle json:chart blocks
                                        if (lang === "json:chart" || lang === "chart") {
                                            return <InlineChart chartJson={codeText} />;
                                        }

                                        // Auto-detect chart JSON
                                        if (lang === "json") {
                                            try {
                                                const parsed = JSON.parse(codeText);
                                                if (parsed.type && parsed.data && Array.isArray(parsed.data)) {
                                                    return <InlineChart chartJson={codeText} />;
                                                }
                                            } catch { }
                                        }

                                        // Regular code block
                                        if (match) {
                                            return <CodeBlock language={lang}>{children}</CodeBlock>;
                                        }

                                        // Inline code
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
                                        <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-primary)" }} className="hover:underline">
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
                                    thead: ({ children }: any) => <thead style={{ background: "var(--bg-tertiary)" }}>{children}</thead>,
                                    tbody: ({ children }: any) => <tbody className="divide-y" style={{ borderColor: "var(--border-secondary)" }}>{children}</tbody>,
                                    tr: ({ children }: any) => <tr className="transition-colors" style={{ borderColor: "var(--border-secondary)" }}>{children}</tr>,
                                    th: ({ children }: any) => (
                                        <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
                                            {children}
                                        </th>
                                    ),
                                    td: ({ children }: any) => (
                                        <td className="px-4 py-3 text-[13px] whitespace-nowrap" style={{ color: "var(--text-primary)" }}>
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
                            <button onClick={handleCopy} className="action-btn tooltip-wrapper" data-tooltip="Copy">
                                {copied ? <Check className="w-4 h-4" style={{ color: "#34a853" }} /> : <Copy className="w-4 h-4" />}
                            </button>
                            {isLatest && onRegenerate && (
                                <button onClick={onRegenerate} className="action-btn tooltip-wrapper" data-tooltip="Regenerate">
                                    <RefreshCw className="w-4 h-4" />
                                </button>
                            )}
                            <button onClick={handleSpeak} className="action-btn tooltip-wrapper" data-tooltip="Read aloud">
                                <Volume2 className="w-4 h-4" />
                            </button>
                            <div className="w-px h-4 mx-1" style={{ background: "var(--border-primary)" }} />
                            <button onClick={() => setLiked(liked === "up" ? null : "up")} className={cn("action-btn", liked === "up" && "liked")}>
                                <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button onClick={() => setLiked(liked === "down" ? null : "down")} className={cn("action-btn", liked === "down" && "liked")}>
                                <ThumbsDown className="w-4 h-4" />
                            </button>
                        </div>
                    )}

                    {/* Structured Data Table + Chart */}
                    {structuredData && structuredData.headers && structuredData.rows && structuredData.rows.length > 0 && (
                        <div className="mt-5 mb-2 space-y-4 message-fade-in">
                            <ChartRenderer data={{ headers: structuredData.headers, rows: structuredData.rows }} />
                            <div className="data-table overflow-x-auto">
                                <div
                                    className="flex items-center justify-between px-4 py-2.5"
                                    style={{ background: "var(--bg-tertiary)", borderBottom: "1px solid var(--border-primary)" }}
                                >
                                    <span className="text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1.5" style={{ color: "var(--text-secondary)" }}>
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
                                                    style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}
                                                >
                                                    {header.replace(/_/g, " ")}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {structuredData.rows.map((row: any[], i: number) => (
                                            <tr key={i} className="transition-colors hover:bg-[var(--bg-secondary)]" style={{ borderTop: "1px solid var(--border-secondary)" }}>
                                                {row.map((cell: any, j: number) => (
                                                    <td key={j} className="px-4 py-3 text-[13px] whitespace-nowrap font-medium" style={{ color: "var(--text-primary)" }}>
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
                <div className="flex-shrink-0 w-8 h-8 mt-1 rounded-full flex items-center justify-center" style={{ background: "var(--accent-primary)" }}>
                    <User className="w-4 h-4 text-white" />
                </div>
            )}
        </div>
    );
}
