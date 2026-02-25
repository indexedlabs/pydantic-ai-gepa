from pydantic_ai_gepa.skills.fs import SkillsFS
from pydantic_ai_gepa.skill_components import (
    apply_candidate_to_skills,
    skill_description_key,
    skill_body_key,
)
from pydantic_ai_gepa.gepa_graph.models.candidate import ComponentValue


def test_create_new_skill():
    base_fs = SkillsFS()
    candidate = {
        skill_description_key("new-skill"): ComponentValue(
            name=skill_description_key("new-skill"), text="A new skill"
        ),
        skill_body_key("new-skill"): ComponentValue(
            name=skill_body_key("new-skill"), text="This is the body."
        ),
    }

    with apply_candidate_to_skills(base_fs, candidate) as view:
        assert view.exists("new-skill/SKILL.md")
        content = view.read_text("new-skill/SKILL.md")
        assert "name: new-skill" in content
        assert "description: A new skill" in content
        assert "This is the body." in content
