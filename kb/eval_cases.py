"""
The evaluation set: twenty cases, one source of truth for both eval harnesses.

Each case carries three views of the same thing, because the two harnesses measure
different tasks:

  query      how an EVENT would describe it -- "worker not wearing a hard hat".
             This is what retrieval actually receives in production, and what
             kb/eval_retrieval.py measures Recall@k against.

  question   the same thing asked as a QUESTION. Ragas grades question-answering, and
             an event description is not a question: a model that correctly declines to
             invent consequences for "worker not wearing a hard hat" scores zero for
             irrelevance. Measuring the wrong task produces numbers that look like a
             system failure and are not.

  reference  a hand-written ground-truth ANSWER, drawn from the clause text. Ragas
             compares retrieved context against this. Passing the whole clause instead
             caps context_recall by construction -- retrieval returns one paragraph and
             the reference holds eight, so it can never look complete.

`expect_section` is the clause kb/clauses.yaml says applies. Both harnesses grade
against that, never against whatever search happened to return.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    query: str
    expect_section: str
    item: str | None = None
    note: str = ""
    question: str = ""
    reference: str = ""

    @property
    def as_question(self) -> str:
        return self.question or self.query


CASES: list[Case] = [
    # --- head -----------------------------------------------------------
    Case("worker not wearing a hard hat", "1926.100", "helmet",
         question="When must employees wear protective helmets on a construction site?",
         reference="Employees must be protected by protective helmets when working in "
                   "areas where there is a possible danger of head injury from impact, "
                   "from falling or flying objects, or from electrical shock and burns."),
    Case("no helmet in an area with overhead work", "1926.100", "helmet",
         question="Is head protection required where objects may fall from above?",
         reference="Yes. Protective helmets are required where there is a possible "
                   "danger of head injury from falling or flying objects."),
    Case("head protection requirements and ANSI standard", "1926.100", "helmet",
         question="Which consensus standards may head protection meet?",
         reference="Head protection must meet ANSI Z89.1-2009, ANSI Z89.1-2003, or "
                   "ANSI Z89.1-1997. Any one of the three is acceptable."),
    Case("bare head near falling objects", "1926.100", "helmet",
         question="What protects an employee from head injury by falling objects?",
         reference="A protective helmet. Employees in areas with a possible danger of "
                   "head injury from falling or flying objects must wear one."),

    # --- eye and face ---------------------------------------------------
    Case("no eye protection while using a grinder", "1926.102", "goggles",
         question="When must eye and face protection be provided?",
         reference="Employees must be provided with eye and face protection equipment "
                   "when machines or operations present potential eye or face injury "
                   "from physical, chemical or radiation agents."),
    Case("worker without goggles near flying particles", "1926.102", "goggles",
         question="Is eye protection required where particles may fly?",
         reference="Yes. Eye and face protection is required when operations present "
                   "potential eye injury from physical agents such as flying particles."),
    Case("face shield requirements for welding", "1926.102", "goggles",
         question="What eye protection applies to welding operations?",
         reference="Employees engaged in welding must use eye protection such as helmets "
                   "or goggles fitted with filter lenses of the shade appropriate to the "
                   "work being performed."),
    Case("eye protection for employees exposed to chemical splash", "1926.102", "goggles",
         question="Is eye protection required against chemical agents?",
         reference="Yes. Eye and face protection is required when operations present "
                   "potential eye or face injury from chemical agents."),

    # --- feet -----------------------------------------------------------
    Case("no safety boots on the loading dock", "1926.96", "boots",
         question="When must safety-toe footwear be worn?",
         reference="1926.96 states only that safety-toe footwear must meet ANSI Z41.1-1967. It does not say when footwear must be worn; that duty comes from 1926.95(a)."),
    Case("foot protection where objects may fall or roll", "1926.96", "boots",
         question="What foot protection applies where objects may fall or roll?",
         reference="Safety-toe footwear must meet ANSI Z41.1-1967 under 1926.96. The requirement to use protective equipment where hazards exist comes from 1926.95(a)."),
    Case("worker in trainers on site", "1926.96", "boots",
         question="Is ordinary footwear acceptable where foot injury is possible?",
         reference="1926.96 specifies that safety-toe footwear must meet ANSI Z41.1-1967. Whether ordinary footwear is acceptable turns on 1926.95(a), which requires protective equipment wherever hazards make it necessary."),

    # --- hands ----------------------------------------------------------
    # Construction has no hand-protection section, so these must land on the general
    # criterion. A hit on 1910.138 would mean the corpus was built from the wrong part.
    Case("worker handling materials with bare hands", "1926.95", "gloves",
         question="What does the construction PPE criterion say about protecting the "
                  "extremities?",
         reference="Protective equipment for the extremities must be provided, used and "
                   "maintained in a sanitary and reliable condition wherever hazards of "
                   "processes or environment make it necessary."),
    Case("no gloves while operating an abrasive wheel", "1926.95", "gloves",
         question="Must hand protection be provided where a process creates a hazard?",
         reference="Yes. Protective equipment for the extremities must be provided and "
                   "used wherever hazards of processes or environment make it necessary "
                   "to protect against injury or impairment."),
    Case("hand protection requirement on a construction site", "1926.95", "gloves",
         question="Who must provide protective equipment for the hands in construction?",
         reference="The employer. Protective equipment for the extremities must be "
                   "provided, used and maintained wherever it is necessary by reason of "
                   "hazards of processes or environment."),

    # --- high visibility ------------------------------------------------
    Case("worker without a high visibility vest", "1926.95", "vest",
         question="Does the general construction PPE criterion cover protective clothing?",
         reference="Yes. Protective clothing must be provided, used and maintained in a "
                   "sanitary and reliable condition wherever hazards of processes or "
                   "environment make it necessary."),
    Case("no hi-vis clothing near moving plant", "1926.95", "vest",
         question="On what basis is protective clothing required where hazards exist?",
         reference="Protective clothing must be provided wherever it is necessary by "
                   "reason of hazards of processes or environment capable of causing "
                   "injury or impairment."),
    Case("flagger directing traffic without a warning garment", "1926.201", "vest",
         note="the one case where hi-vis IS squarely regulated",
         question="What must a flagger wear while directing traffic?",
         reference="Flaggers must be provided with and must wear a warning garment such "
                   "as a vest, jacket or shirt. Garments worn at night must be "
                   "retroreflective."),

    # --- general duty ---------------------------------------------------
    Case("who is responsible for providing protective equipment", "1926.28",
         question="Who is responsible for requiring the use of PPE on a construction site?",
         reference="The employer is responsible for requiring the wearing of appropriate "
                   "personal protective equipment in all operations where there is an "
                   "exposure to hazardous conditions."),
    Case("employer duty to require PPE where hazards exist", "1926.28",
         question="What is the employer's duty where employees are exposed to hazards?",
         reference="The employer must require the wearing of appropriate personal "
                   "protective equipment in all operations where there is an exposure to "
                   "hazardous conditions, or where this part indicates the need for it."),
    Case("protective equipment shall be provided and maintained in sanitary condition",
         "1926.95",
         question="In what condition must protective equipment be maintained?",
         reference="Protective equipment must be maintained in a sanitary and reliable "
                   "condition wherever it is necessary by reason of hazards of processes "
                   "or environment."),
]
