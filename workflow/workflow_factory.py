from langgraph.graph import END

from workflow.graph_builder import GraphBuilder


class WorkflowFactory:
    """Factory for creating appropriate workflows"""

    @staticmethod
    def translate_only_workflow():
        """Create a workflow that only translates the transcript"""
        builder = GraphBuilder()
        builder.add_context_translator("context_translate")
        builder.workflow.add_edge("context_translate", END)
        return builder.build()

    @staticmethod
    def translate_and_create_subtitles_workflow():
        builder = GraphBuilder()
        builder.add_transcriber("transcriber")
        builder.add_summarizer("summarizer")
        builder.add_whisper_translator("whisper_translate")
