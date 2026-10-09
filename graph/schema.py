from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    DRUG = "Drug"
    CONDITION = "Condition"
    SIDE_EFFECT = "SideEffect"


class RelationType(str, Enum):
    TREATS = "TREATS"
    INTERACTS_WITH = "INTERACTS_WITH"
    CAUSES = "CAUSES"
    CONTRAINDICATED_IN = "CONTRAINDICATIONS_IN"


class ClinicalEntity(BaseModel):
    name: str = Field(description="Normalized name of the clinical entity, e.g., 'Warfarin'")
    entity_type: EntityType = Field(description="Category of the entity (Drug, Condition, SideEffect)")


class ClinicalRelationship(BaseModel):
    source: str = Field(description="Name of source entity")
    relation: RelationType = Field(description="Standardized relationship type")
    target: str = Field(description="Name of target entity")
    evidence_snippet: Optional[str] = Field(default=None, description="Brief snippet supporting the relationship")


class ExtractedClinicalTriples(BaseModel):
    entities: List[ClinicalEntity] = Field(default_factory=list)
    relationships: List[ClinicalRelationship] = Field(default_factory=list)


if __name__ == "__main__":
    # Test schema validation
    sample = ExtractedClinicalTriples(
        entities=[
            ClinicalEntity(name="Warfarin", entity_type=EntityType.DRUG),
            ClinicalEntity(name="Atrial Fibrillation", entity_type=EntityType.CONDITION),
            ClinicalEntity(name="Bleeding", entity_type=EntityType.SIDE_EFFECT)
        ],
        relationships=[
            ClinicalRelationship(source="Warfarin", relation=RelationType.TREATS, target="Atrial Fibrillation"),
            ClinicalRelationship(source="Warfarin", relation=RelationType.CAUSES, target="Bleeding")
        ]
    )

    print("[✓] Clinical Schema validated successfully!")
    print(f"Entities: {[e.name for e in sample.entities]}")
    print(f"Triples:  {[(r.source, r.relation.value, r.target) for r in sample.relationships]}")