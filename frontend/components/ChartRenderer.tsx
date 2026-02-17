"use client";

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
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";
import { BarChart3, TrendingUp, PieChart as PieIcon } from "lucide-react";

const COLORS = [
    "#4285f4",
    "#34a853",
    "#fbbc05",
    "#ea4335",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#ec4899",
];

const GRADIENTS = [
    { id: "g1", start: "#4285f4", end: "#669df6" },
    { id: "g2", start: "#34a853", end: "#5bb974" },
    { id: "g3", start: "#fbbc05", end: "#f9d56e" },
    { id: "g4", start: "#ea4335", end: "#f28b82" },
    { id: "g5", start: "#a855f7", end: "#c084fc" },
];

interface ChartData {
    headers: string[];
    rows: any[][];
    chartType?: "bar" | "line" | "pie" | "area";
}

interface ChartRendererProps {
    data: ChartData;
}

export default function ChartRenderer({ data }: ChartRendererProps) {
    if (!data || !data.headers || !data.rows || data.rows.length === 0) {
        return null;
    }

    // Transform data for Recharts
    const chartData = data.rows.map((row) => {
        const obj: any = {};
        data.headers.forEach((header, i) => {
            obj[header] = row[i];
        });
        return obj;
    });

    // Determine columns
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

    // Auto-detect chart type
    const chartType =
        data.chartType ||
        (data.rows.length <= 5 ? "bar" : data.rows.length <= 3 ? "pie" : "area");

    const ChartIcon =
        chartType === "line" || chartType === "area"
            ? TrendingUp
            : chartType === "pie"
                ? PieIcon
                : BarChart3;

    const tooltipStyle = {
        contentStyle: {
            background: "var(--bg-primary)",
            border: "1px solid var(--border-primary)",
            borderRadius: "10px",
            boxShadow: "var(--shadow-lg)",
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

    return (
        <div className="chart-container">
            <div
                className="text-[11px] font-semibold uppercase tracking-wider mb-4 flex items-center gap-2"
                style={{ color: "var(--text-secondary)" }}
            >
                <ChartIcon className="w-3.5 h-3.5" style={{ color: "var(--accent-primary)" }} />
                Data Visualization
            </div>
            <ResponsiveContainer width="100%" height={300}>
                {chartType === "bar" ? (
                    <BarChart
                        data={chartData}
                        margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
                    >
                        <defs>
                            {GRADIENTS.map((g) => (
                                <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={g.start} stopOpacity={0.9} />
                                    <stop offset="100%" stopColor={g.end} stopOpacity={0.6} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis
                            dataKey={labelColumn}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <YAxis
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <Tooltip {...tooltipStyle} />
                        <Legend
                            wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }}
                        />
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
                    <AreaChart
                        data={chartData}
                        margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
                    >
                        <defs>
                            {GRADIENTS.map((g) => (
                                <linearGradient key={`area-${g.id}`} id={`area-${g.id}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={g.start} stopOpacity={0.3} />
                                    <stop offset="100%" stopColor={g.end} stopOpacity={0.05} />
                                </linearGradient>
                            ))}
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis
                            dataKey={labelColumn}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <YAxis
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) => (
                            <Area
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={GRADIENTS[idx % GRADIENTS.length].start}
                                fill={`url(#area-${GRADIENTS[idx % GRADIENTS.length].id})`}
                                strokeWidth={2}
                            />
                        ))}
                    </AreaChart>
                ) : chartType === "line" ? (
                    <LineChart
                        data={chartData}
                        margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                        <XAxis
                            dataKey={labelColumn}
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <YAxis
                            tick={{ fontSize: 11, fill: "var(--text-tertiary)" }}
                            axisLine={{ stroke: "var(--border-primary)" }}
                            tickLine={{ stroke: "var(--border-primary)" }}
                        />
                        <Tooltip {...tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {valueColumns.map((col, idx) => (
                            <Line
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2.5}
                                dot={{
                                    fill: COLORS[idx % COLORS.length],
                                    strokeWidth: 2,
                                    r: 4,
                                    stroke: "var(--bg-primary)",
                                }}
                                activeDot={{
                                    r: 6,
                                    stroke: COLORS[idx % COLORS.length],
                                    strokeWidth: 2,
                                    fill: "var(--bg-primary)",
                                }}
                            />
                        ))}
                    </LineChart>
                ) : (
                    <PieChart>
                        <defs>
                            {COLORS.map((color, i) => (
                                <linearGradient key={`pie-${i}`} id={`pie-${i}`} x1="0" y1="0" x2="1" y2="1">
                                    <stop offset="0%" stopColor={color} stopOpacity={0.9} />
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
                            outerRadius={100}
                            innerRadius={40}
                            paddingAngle={2}
                            label={({ name, percent }) =>
                                `${name} ${(percent * 100).toFixed(0)}%`
                            }
                            labelLine={{ stroke: "var(--text-tertiary)" }}
                        >
                            {chartData.map((_, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={`url(#pie-${index % COLORS.length})`}
                                    stroke="var(--bg-primary)"
                                    strokeWidth={2}
                                />
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
