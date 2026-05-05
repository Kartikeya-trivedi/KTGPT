import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const MODAL_URL = "http://localhost:8000/chat";

    const response = await fetch(MODAL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: body.prompt,
        toolMode: body.toolMode ?? false,
      }),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("API Route Error:", error);
    return NextResponse.json({ error: "Failed to connect to backend. Is local_service.py running?" }, { status: 500 });
  }
}
