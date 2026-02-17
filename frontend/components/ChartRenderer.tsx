"use client";

import { useState, useEffect } from "react";
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
    AreaChart,
    Area,
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
import {
    BarChart3,
    TrendingUp,
    PieChart as PieIcon,
    Activity,
    Target,
    Radar as RadarIcon,
    Maximize2,
} from "lucide-react";

const COLORS = [
    "#4285f4",
    "#34a853",
    "#fbbc05",
    "#ea4335",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#ec4899",
    "#14b8a6",
    "#8b5cf6",
];

const GRADIENTS = [
    { id: "cg1", start: "#4285f4", end: "#1a73e8" },
    { id: "cg2", start: "#34a853", end: "#1e8e3e" },
    { id: "cg3", start: "#fbbc05", end: "#f9ab00" },
    { id: "cg4", start: "#ea4335", end: "#d93025" },
    { id: "cg5", start: "#a855f7", end: "#7c3aed" },
    { id: "cg6", start: "#06b6d4", end: "#0891b2" },
    { id: "cg7", start: "#f97316", end: "#ea580c" },
    { id: "cg8", start: "#ec4899", end: "#db2777" },
];

interface ChartData {
    headers: string[];
    rows: any[][];
    chartType?: string;
}

interface ChartRendererProps {
    data: ChartData;
}

const tooltipStyle = {
    contentStyle: {
        background: "var(--bg-primary)",
        border: "1px solid var(--border-primary)",
        borderRadius: "12px",
        boxShadow: "var(--shadow-lg)",
        color: "var(--text-primary)",
        fontSize: 12,
        padding: "12px 16px",
    },
    labelStyle: {
        color: "var(--text-secondary)",
        fontWeight: 600,
        marginBottom: 4,
    },
};

export default function ChartRenderer({ data }: ChartRendererProps) {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    if (!mounted || !data?.headers?.length || !data?.rows?.length) return null;

    // Transform data
    const chartData = data.rows.map((row) => {
        const obj: any = {};
        data.headers.forEach((header, i) => {
            obj[header] = row[i];
        });
        return obj;
    });

    const numericColumns = data.headers.filter((_, i) =>
        data.rows.every(
            (row) => typeof row[i] === "number" || !isNaN(parseFloat(row[i]))
        )
    );

    const labelColumn =
        data.headers.find((_, i) =>
            data.rows.some(
                (row) => typeof row[i] === "string" && isNaN(parseFloat(row[i]))
            )
        ) || data.headers[0];

    const valueColumns =
        numericColumns.length > 0 ? numericColumns : [data.headers[1]];

    // Auto-detect best chart type
    const chartType =
        data.chartType ||
        (valueColumns.length >= 3
            ? "composed"
            : data.rows.length <= 6
                ? "bar"
                : "area");

    const ChartIcon =
        chartType === "line" || chartType === "area"
            ? TrendingUp
            : chartType === "pie"
                ? PieIcon
                : chartType === "radar"
                    ? Target
                    : chartType === "scatter"
                        ? Activity
                        : BarChart3;

    return (
        <div className="chart-container">
            {/* Chart Header */}
            <div
                className="flex items-center justify-between mb-4"
                style={{ borderBottom: "1px solid var(--border-secondary)", paddingBottom: 10 }}
            >
                <div className="flex items-center gap-2">
                    <div
                        className="w-7 h-7 rounded-lg flex items-center justify-center"
                        style={{ background: "var(--accent-primary)", opacity: 0.15 }}
                    >
                        <ChartIcon className="w-3.5 h-3.5" style={{ color: "var(--accent-primary)" }} />
                    </div>
                    <span
                        className="text-[11px] font-bold uppercase tracking-wider"
                        style={{ color: "var(--text-secondary)" }}
                    >
                        Data Visualization
                    </span>
                </div>
                <span
                    className="text-[10px] font-medium px-2 py-1 rounded-full"
                    style={{
                        background: "var(--bg-tertiary)",
                        color: "var(--text-tertiary)",
                        border: "1px solid var(--border-primary)",
                    }}
                >
                    {chartType.toUpperCase()} • {data.rows.length} records
                </span>
            </div>

            {/* Chart Body */}
            <ResponsiveContainer width="100%" height={320}>
                {chartType === "bar" ? (
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                            {GRADIENTS.map((g) => (
                                <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={g.start} stopOpacity={0.9} />
                                    <stop offset="100%" stopColor={g.end} stopOpacity={0.5} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis
                            dataKey={labelColumn}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                        />
                        <YAxis tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) => (
                            <Bar
                                key={col}
                                dataKey={col}
                                fill={`url(#${GRADIENTS[idx % GRADIENTS.length].id})`}
                                radius={[6, 6, 0, 0]}
                            />
                        ))}
                    </BarChart>
                ) : chartType === "area" ? (
                    <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                            {GRADIENTS.map((g) => (
                                <linearGradient key={`a-${g.id}`} id={`a-${g.id}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={g.start} stopOpacity={0.35} />
                                    <stop offset="100%" stopColor={g.end} stopOpacity={0.05} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis dataKey={labelColumn} tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <YAxis tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) => (
                            <Area
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={GRADIENTS[idx % GRADIENTS.length].start}
                                fill={`url(#a-${GRADIENTS[idx % GRADIENTS.length].id})`}
                                strokeWidth={2.5}
                            />
                        ))}
                    </AreaChart>
                ) : chartType === "line" ? (
                    <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis dataKey={labelColumn} tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <YAxis tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) => (
                            <Line
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2.5}
                                dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 2, r: 4, stroke: "var(--bg-primary)" }}
                                activeDot={{ r: 6, stroke: COLORS[idx % COLORS.length], strokeWidth: 2, fill: "var(--bg-primary)" }}
                            />
                        ))}
                    </LineChart>
                ) : chartType === "composed" ? (
                    <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                            {GRADIENTS.map((g) => (
                                <linearGradient key={`comp-${g.id}`} id={`comp-${g.id}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={g.start} stopOpacity={0.8} />
                                    <stop offset="100%" stopColor={g.end} stopOpacity={0.4} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis dataKey={labelColumn} tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        {valueColumns.length > 1 && (
                            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "var(--text-tertiary)" }} axisLine={{ stroke: "var(--border-primary)" }} />
                        )}
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) =>
                            idx === 0 ? (
                                <Bar
                                    key={col}
                                    yAxisId="left"
                                    dataKey={col}
                                    fill={`url(#comp-${GRADIENTS[idx % GRADIENTS.length].id})`}
                                    radius={[6, 6, 0, 0]}
                                />
                            ) : (
                                <Line
                                    key={col}
                                    yAxisId={idx === 1 && valueColumns.length > 1 ? "right" : "left"}
                                    type="monotone"
                                    dataKey={col}
                                    stroke={COLORS[idx % COLORS.length]}
                                    strokeWidth={2.5}
                                    dot={{ fill: COLORS[idx % COLORS.length], r: 4, stroke: "var(--bg-primary)", strokeWidth: 2 }}
                                />
                            )
                        )}
                    </ComposedChart>
                ) : chartType === "radar" ? (
                    <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="75%">
                        <PolarGrid stroke="var(--border-primary)" />
                        <PolarAngleAxis dataKey={labelColumn} tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} />
                        <PolarRadiusAxis tick={{ fontSize: 9, fill: "var(--text-tertiary)" }} />
                        {valueColumns.map((col, idx) => (
                            <Radar
                                key={col}
                                name={col}
                                dataKey={col}
                                stroke={COLORS[idx % COLORS.length]}
                                fill={COLORS[idx % COLORS.length]}
                                fillOpacity={0.15}
                                strokeWidth={2}
                            />
                        ))}
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                    </RadarChart>
                ) : chartType === "scatter" ? (
                    <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis
                            dataKey={valueColumns[0] || labelColumn}
                            name={valueColumns[0] || labelColumn}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                        />
                        <YAxis
                            dataKey={valueColumns[1] || valueColumns[0]}
                            name={valueColumns[1] || valueColumns[0]}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                        />
                        <ZAxis range={[60, 300]} />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Scatter
                            name="Data Points"
                            data={chartData}
                            fill={COLORS[0]}
                            fillOpacity={0.7}
                            strokeWidth={1}
                            stroke={COLORS[0]}
                        />
                    </ScatterChart>
                ) : (
                    <PieChart>
                        <defs>
                            {COLORS.map((color, i) => (
                                <linearGradient key={`pie-${i}`} id={`pie-${i}`} x1="0" y1="0" x2="1" y2="1">
                                    <stop offset="0%" stopColor={color} stopOpacity={0.95} />
                                    <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                                </linearGradient>
                            ))}
                        </defs>
                        <Pie
                            data={chartData}
                            dataKey={valueColumns[0]}
                            nameKey={labelColumn}
                            cx="50%"
                            cy="50%"
                            outerRadius={110}
                            innerRadius={50}
                            paddingAngle={3}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            labelLine={{ stroke: "var(--text-tertiary)" }}
                        >
                            {chartData.map((_, index) => (
                                <Cell key={`cell-${index}`} fill={`url(#pie-${index % COLORS.length})`} stroke="var(--bg-primary)" strokeWidth={2} />
                            ))}
                        </Pie>
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                    </PieChart>
                )}
            </ResponsiveContainer>
        </div>
    );
}
