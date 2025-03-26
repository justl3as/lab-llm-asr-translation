from langgraph.graph import END

from workflow.graph_builder import GraphBuilder


class WorkflowFactory:
    """Factory for creating appropriate workflows"""

    @staticmethod
    def transcribe_summarize_translate_workflow():
        """Create a workflow that transcribes, summarizes, and translates"""
        builder = GraphBuilder()
        builder.add_audio_extractor("audio_extractor")
        builder.add_transcriber("transcriber")
        builder.add_summarizer("summarizer")
        builder.add_whisper_translator("whisper_translate")

        builder.add_edge("audio_extractor", "transcriber")
        builder.add_edge("transcriber", "summarizer")
        builder.add_edge("summarizer", "whisper_translate")
        builder.add_edge("whisper_translate", END)

        return builder.build()
