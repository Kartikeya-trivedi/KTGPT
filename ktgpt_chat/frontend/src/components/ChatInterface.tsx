"use client";

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  tool_call?: any;
  tool_result?: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [toolMode, setToolMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: currentInput, toolMode }),
      });

      const data = await response.json();

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer || data.raw_output || "No response received.",
        tool_call: data.tool_call,
        tool_result: data.tool_result
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Failed to fetch:", error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Failed to connect to the backend. Is local_service.py running?"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestion = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  return (
    <div className="relative min-h-screen">
      {/* Ambient background */}
      <div className="ambient-bg" />

      <div className="relative z-10 flex flex-col h-screen max-w-3xl mx-auto px-4 py-5">

        {/* ── Header ── */}
        <header className="glass rounded-2xl px-5 py-4 mb-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--accent)] to-[#4834d4] flex items-center justify-center glow-accent">
              <span className="text-white font-bold text-sm mono">KT</span>
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                KT-GPT <span className="gradient-text font-bold">1B</span>
              </h1>
              <p className="text-[11px] text-[var(--text-muted)] tracking-wide">
                Sparse MoE • MLA • Local GPU
              </p>
            </div>
          </div>

          {/* Tool Mode Toggle */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-[var(--text-secondary)]">
              {toolMode ? '🔧 Tools' : '💬 Chat'}
            </span>
            <div
              className={`toggle-track ${toolMode ? 'active' : ''}`}
              onClick={() => setToolMode(!toolMode)}
              title={toolMode ? 'Tool mode ON — calculator & search available' : 'Chat mode — direct conversation'}
            >
              <div className="toggle-thumb" />
            </div>
          </div>
        </header>

        {/* Tool mode indicator */}
        <AnimatePresence>
          {toolMode && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-3 overflow-hidden"
            >
              <div className="tool-badge rounded-xl px-4 py-2.5 flex items-center gap-3 text-xs">
                <span className="text-[var(--green)] font-medium">⚡ Tool Mode Active</span>
                <span className="text-[var(--text-muted)]">•</span>
                <span className="text-[var(--text-secondary)]">calculator</span>
                <span className="text-[var(--text-muted)]">•</span>
                <span className="text-[var(--text-secondary)]">search</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Chat Messages ── */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-1">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center gap-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[var(--accent)]/10 to-transparent border border-[var(--accent)]/20 flex items-center justify-center">
                <span className="text-3xl">🧠</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-1">KT-GPT is ready</h2>
                <p className="text-sm text-[var(--text-muted)]">
                  {toolMode
                    ? 'Ask a math question — the calculator tool will handle it.'
                    : 'Ask me anything. Toggle tools for calculator & search.'}
                </p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center max-w-md">
                {toolMode ? (
                  <>
                    <button onClick={() => handleSuggestion('What is 347 * 891?')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      What is 347 × 891?
                    </button>
                    <button onClick={() => handleSuggestion('Calculate the square root of 2025')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      √2025
                    </button>
                    <button onClick={() => handleSuggestion('Who won the Nobel Prize in 2023?')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      Nobel Prize 2023?
                    </button>
                  </>
                ) : (
                  <>
                    <button onClick={() => handleSuggestion('Say hello in French')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      Say hello in French
                    </button>
                    <button onClick={() => handleSuggestion('Explain how gravity works')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      Explain gravity
                    </button>
                    <button onClick={() => handleSuggestion('What is machine learning?')} className="chip rounded-full px-4 py-2 text-xs text-[var(--text-secondary)]">
                      What is ML?
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className="flex flex-col gap-2 max-w-[85%]">
                  {/* Label */}
                  <span className={`text-[10px] uppercase tracking-widest font-medium ${
                    msg.role === 'user' ? 'text-right text-[var(--accent-light)]' : 'text-[var(--text-muted)]'
                  }`}>
                    {msg.role === 'user' ? 'You' : 'KT-GPT'}
                  </span>

                  {/* Bubble */}
                  <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === 'user' ? 'msg-user' : 'msg-assistant'
                  }`}>
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>

                  {/* Tool call badge */}
                  {msg.tool_call && (
                    <div className="tool-badge rounded-xl px-3 py-2 flex items-center gap-2">
                      <span className="text-[var(--green)] text-xs">🔧</span>
                      <span className="text-xs text-[var(--green)] font-medium mono">
                        {msg.tool_call.name}
                      </span>
                      {msg.tool_call.arguments?.expression && (
                        <span className="text-xs text-[var(--text-muted)] mono">
                          ({msg.tool_call.arguments.expression})
                        </span>
                      )}
                      {msg.tool_result && (
                        <>
                          <span className="text-[var(--text-muted)]">→</span>
                          <span className="text-xs text-[var(--orange)] font-medium mono">
                            {msg.tool_result}
                          </span>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Loading indicator */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="msg-assistant rounded-2xl px-5 py-4 flex items-center gap-1.5">
                <div className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                <div className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                <div className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ── Input ── */}
        <form onSubmit={handleSubmit} className="mt-4 input-area rounded-2xl flex items-center gap-2 p-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={toolMode ? 'Ask a math question...' : 'Send a message...'}
            className="flex-1 bg-transparent border-none outline-none text-sm px-3 py-2.5 text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-[var(--accent)] hover:bg-[#5a4bd4] disabled:opacity-30 disabled:cursor-not-allowed
                       px-4 py-2.5 rounded-xl text-white text-sm font-medium transition-all duration-200
                       hover:shadow-lg hover:shadow-[var(--accent-glow)]"
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </form>

        <p className="text-[9px] text-center text-[var(--text-muted)] mt-2.5 tracking-widest uppercase opacity-40">
          KT-GPT • 1B Parameters • Sparse MoE + MLA • Local Inference
        </p>
      </div>
    </div>
  );
}
