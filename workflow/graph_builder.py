from langgraph.graph import StateGraph

from processors.audio_extractor import AudioExtractor
from processors.summarizer import Summarizer
from processors.transcriber import TranscribeAudio
from processors.translator import (
    ContextTranslator,
    WhisperTranslator,
)
from workflow.state import State


class GraphBuilder:
    """Builder Pattern for creating LangGraph workflow"""

    def __init__(self):
        self.workflow = StateGraph(State)
        self.entry_point = None

    def add_audio_extractor(self):
        """Add audio extractor to the workflow"""
        audio_extractor = AudioExtractor()
        self.workflow.add_node(
            "extract_audio", lambda state: audio_extractor.process(state)
        )
        return self

    def add_transcriber(self, node_name):
        """Add transcriber to the workflow"""
        transcriber = TranscribeAudio()
        self.workflow.add_node(node_name, lambda state: transcriber.process(state))
        return self

    def add_summarizer(self, node_name):
        """Add summarizer to the workflow"""
        summarizer = Summarizer()
        self.workflow.add_node(node_name, lambda state: summarizer.process(state))
        return self

    def add_whisper_translator(self, node_name):
        """Add whisper translator to the workflow"""
        translator = WhisperTranslator()
        self.workflow.add_node(node_name, lambda state: translator.process(state))
        return self

    def add_context_translator(self, node_name):
        """Add context translator to the workflow"""
        translator = ContextTranslator()
        self.workflow.add_node(node_name, lambda state: translator.process(state))
        return self

    def add_edge(self, start_node, end_node):
        """Add an edge between two nodes in the workflow"""
        self.workflow.add_edge(start_node, end_node)

        if self.entry_point is None:
            self.entry_point = start_node
            self.workflow.set_entry_point(self.entry_point)
        return self

    def build(self):
        """Build and return the compiled workflow"""
        return self.workflow.compile()
