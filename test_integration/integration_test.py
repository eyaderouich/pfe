import json
import pytest
from main import VehicleCounter

@pytest.mark.integration
def test_process_video_integration(capsys):
    video_path = "demo1.mp4"
    vc = VehicleCounter(video_path)
    vc.process_video()
    captured = capsys.readouterr()
    json_outputs = captured.out.splitlines()
    for json_output in json_outputs:
        try:
            output_data = json.loads(json_output)
            
            assert "activity" in output_data
            assert "class" in output_data
            assert "classificators" in output_data
            assert "registration" in output_data

            assert output_data["activity"] == "Monitoring"
            assert output_data["class"] == "car"
            assert isinstance(output_data["classificators"], list)
            assert len(output_data["classificators"]) == 1  
            classificator = output_data["classificators"][0]
            assert "make" in classificator
            assert "model" in classificator
            assert "class" in classificator
            assert "color" in classificator
            assert "country" in classificator
            assert "registration" in classificator
        except json.JSONDecodeError:
            pass