import json
from logging.config import fileConfig
from pathlib import Path
from typing import Tuple

from flask import Flask, request, jsonify, Response

class Server:

    def __init__(self):
        fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
        self.app = Flask(__name__)
        self.engine = AnalyzerEngine()

        @self.app.route("/analyze", methods=["POST"])
        def analyze() -> Tuple[str, int]:
            try:
                req_data = AnalyzerRequest(request.get_json())
                if not req_data.text:
                    raise Exception("No text provided")

                if not req_data.language:
                    raise Exception("No language provided")

                recognizer_result_list = self.engine.analyze(
                    text=req_data.text,
                    language=req_data.language,
                    correlation_id=req_data.correlation_id,
                    score_threshold=req_data.score_threshold,
                    entities=req_data.entities,
                    return_decision_process=req_data.return_decision_process,
                    ad_hoc_recognizers=req_data.ad_hoc_recognizers,
                    context=req_data.context,
                )

                return Response(
                    json.dumps(
                        recognizer_result_list,
                        default=lambda o: o.to_dict(),
                        sort_keys=True,
                    ),
                    content_type="application/json",
                )
