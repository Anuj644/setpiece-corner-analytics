from setpiece.cli import main


def test_help_lists_all_stages(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    for stage in ["fetch", "ingest", "qa", "rq1", "forecast", "rq2", "rq3",
                  "coverage", "dossier", "paper-assets", "all", "review-export"]:
        assert stage in out

def test_unimplemented_stage_returns_2(capsys):
    assert main(["coverage"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
