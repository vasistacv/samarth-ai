"use client";

import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];

interface ChartData {
    headers: string[];
    rows: any[][];
    chartType?: 'bar' | 'line' | 'pie';
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

    // Determine chart type based on data
    const numericColumns = data.headers.filter((_, i) =>
        data.rows.every(row => typeof row[i] === 'number' || !isNaN(parseFloat(row[i])))
    );

    const labelColumn = data.headers.find((_, i) =>
        data.rows.some(row => typeof row[i] === 'string' && isNaN(parseFloat(row[i])))
    ) || data.headers[0];

    const valueColumns = numericColumns.length > 0 ? numericColumns : [data.headers[1]];

    // Use bar chart by default for comparison data
    const chartType = data.chartType || (data.rows.length <= 5 ? 'bar' : 'line');

    return (
        <div className="w-full my-6 p-4 bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="text-xs font-medium text-gray-500 mb-4 uppercase tracking-wider">
                📊 Data Visualization
            </div>
            <ResponsiveContainer width="100%" height={300}>
                {chartType === 'bar' ? (
                    <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey={labelColumn} tick={{ fontSize: 12 }} stroke="#888" />
                        <YAxis tick={{ fontSize: 12 }} stroke="#888" />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'white',
                                border: '1px solid #e5e7eb',
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                            }}
                        />
                        <Legend />
                        {valueColumns.map((col, idx) => (
                            <Bar key={col} dataKey={col} fill={COLORS[idx % COLORS.length]} radius={[4, 4, 0, 0]} />
                        ))}
                    </BarChart>
                ) : chartType === 'line' ? (
                    <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey={labelColumn} tick={{ fontSize: 12 }} stroke="#888" />
                        <YAxis tick={{ fontSize: 12 }} stroke="#888" />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'white',
                                border: '1px solid #e5e7eb',
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                            }}
                        />
                        <Legend />
                        {valueColumns.map((col, idx) => (
                            <Line
                                key={col}
                                type="monotone"
                                dataKey={col}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2}
                                dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 2 }}
                            />
                        ))}
                    </LineChart>
                ) : (
                    <PieChart>
                        <Pie
                            data={chartData}
                            dataKey={valueColumns[0]}
                            nameKey={labelColumn}
                            cx="50%"
                            cy="50%"
                            outerRadius={100}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                            {chartData.map((_, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                    </PieChart>
                )}
            </ResponsiveContainer>
        </div>
    );
}
