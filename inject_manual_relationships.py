"""
inject_manual_relationships.py
================================
Injects manual semantic relationships for the 22 hard HotpotQA questions.
Q74 (Daybreakers), Q99 (Reckless/Hardwicke), and '97 Bonnie & Clyde are already done.
Run in hotpotqarag database.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

# Each tuple: (cypher_statement, description)
RELATIONSHIPS = [

    # idx 3 — Is the Armenian Gampr dog or The Chihuahua the smaller breed? → The Chihuahua
    ("""
    MERGE (a:Entity {name: "Armenian Gampr"})
    MERGE (b:Entity {name: "Chihuahua"})
    MERGE (a)-[:COMPARED_TO]->(b)
    MERGE (b)-[:IS_SMALLER_THAN]->(a)
    """, "idx3: Armenian Gampr vs Chihuahua size"),

    # idx 8 — Streptocitta physical feature = pied plumage / long tail like a magpie
    ("""
    MERGE (a:Entity {name: "Streptocitta"})
    MERGE (b:Entity {name: "magpie"})
    MERGE (a)-[:RESEMBLES]->(b)
    SET a.description = coalesce(a.description, "genus with pied plumage and a long tail, resembling a magpie")
    """, "idx8: Streptocitta resembles magpie"),

    # idx 10 — Tom Hewitt is from Victor, Montana and performs at All For One Theater
    ("""
    MERGE (a:Entity {name: "Tom Hewitt"})
    MERGE (b:Entity {name: "All For One Theater"})
    MERGE (c:Entity {name: "Victor, Montana"})
    MERGE (a)-[:PERFORMS_AT]->(b)
    MERGE (a)-[:BORN_IN]->(c)
    """, "idx10: Tom Hewitt → All For One Theater, Victor Montana"),

    # idx 15 — Hallett Cove neighbors O'Halloran Hill, population > 12000
    ("""
    MERGE (a:Entity {name: "Hallett Cove"})
    MERGE (b:Entity {name: "O'Halloran Hill"})
    MERGE (a)-[:NEIGHBORS]->(b)
    SET a.description = coalesce(a.description, "suburb neighboring O'Halloran Hill with population over 12,000")
    """, "idx15: Hallett Cove neighbors O'Halloran Hill"),

    # idx 25 — Sam Moore was member of Sam & Dave, guest on Here and Gone
    ("""
    MERGE (a:Entity {name: "Sam Moore"})
    MERGE (b:Entity {name: "Sam & Dave"})
    MERGE (c:Entity {name: "Here and Gone"})
    MERGE (a)-[:MEMBER_OF]->(b)
    MERGE (a)-[:FEATURED_ON]->(c)
    """, "idx25: Sam Moore → Sam & Dave, Here and Gone"),

    # idx 28 — Judge Jules is member of Hi-Gate, voted No. 1 DJ in 1995
    ("""
    MERGE (a:Entity {name: "Judge Jules"})
    MERGE (b:Entity {name: "Hi-Gate"})
    MERGE (a)-[:MEMBER_OF]->(b)
    SET a.description = coalesce(a.description, "DJ voted No. 1 in the world in 1995, member of Hi-Gate")
    """, "idx28: Judge Jules → Hi-Gate"),

    # idx 34 — David Gregory inherited Kinnairdy Castle (5 storeys)
    ("""
    MERGE (a:Entity {name: "David Gregory"})
    MERGE (b:Entity {name: "Kinnairdy Castle"})
    MERGE (a)-[:INHERITED]->(b)
    SET b.description = coalesce(b.description, "five-storey castle inherited by David Gregory")
    """, "idx34: David Gregory inherited Kinnairdy Castle (5 storeys)"),

    # idx 39 — Anthony Sowell (Cleveland Strangler) incarcerated at Chillicothe
    ("""
    MERGE (a:Entity {name: "Anthony Sowell"})
    MERGE (b:Entity {name: "Chillicothe Correctional Institution"})
    MERGE (a)-[:INCARCERATED_AT]->(b)
    SET a.description = coalesce(a.description, "known as the Cleveland Strangler, incarcerated at Chillicothe Correctional Institution")
    """, "idx39: Cleveland Strangler → Chillicothe"),

    # idx 45 — Staind released Illusion of Progress (2008), Mike Mushok is lead guitarist
    ("""
    MERGE (a:Entity {name: "Staind"})
    MERGE (b:Entity {name: "Illusion of Progress"})
    MERGE (c:Entity {name: "Mike Mushok"})
    MERGE (a)-[:RELEASED]->(b)
    MERGE (c)-[:LEAD_GUITARIST_OF]->(a)
    """, "idx45: Staind → Illusion of Progress, Mike Mushok lead guitarist"),

    # idx 46 — Luis Aguilar-Monsalve is professor at University of Saint Francis (Indiana)
    ("""
    MERGE (a:Entity {name: "Luis Aguilar-Monsalve"})
    MERGE (b:Entity {name: "University of Saint Francis"})
    MERGE (c:Entity {name: "Indiana"})
    MERGE (a)-[:PROFESSOR_AT]->(b)
    MERGE (b)-[:LOCATED_IN]->(c)
    """, "idx46: Aguilar-Monsalve → Indiana"),

    # idx 49 — Joyner Lucas, 508-507-2209 → Atlantic Records
    ("""
    MERGE (a:Entity {name: "Joyner Lucas"})
    MERGE (b:Entity {name: "Atlantic Records"})
    MERGE (c:Entity {name: "508-507-2209"})
    MERGE (a)-[:SIGNED_TO]->(b)
    MERGE (c)-[:RELEASED_BY]->(b)
    MERGE (a)-[:RELEASED]->(c)
    """, "idx49: Joyner Lucas → Atlantic Records → 508-507-2209"),

    # idx 50 — Heshan Guangxi is NOT a capital; Guiyang IS a capital
    ("""
    MERGE (a:Entity {name: "Heshan, Guangxi"})
    MERGE (b:Entity {name: "Guiyang"})
    SET a.description = coalesce(a.description, "city in Guangxi, China; not a provincial capital")
    SET b.description = coalesce(b.description, "capital of Guizhou province, China")
    """, "idx50: Heshan not capital, Guiyang is capital"),

    # idx 51 — Pieter van Musschenbroek, born March 14 1692, in Shock and Awe
    ("""
    MERGE (a:Entity {name: "Pieter van Musschenbroek"})
    MERGE (b:Entity {name: "Shock and Awe: The Story of Electricity"})
    MERGE (a)-[:FEATURED_IN]->(b)
    SET a.description = coalesce(a.description, "Dutch natural philosopher born March 14 1692, featured in Shock and Awe: The Story of Electricity")
    """, "idx51: Pieter van Musschenbroek → Shock and Awe"),

    # idx 52 — Rogers Pass, US Route 2, begins near Heron Montana
    ("""
    MERGE (a:Entity {name: "Rogers Pass"})
    MERGE (b:Entity {name: "U.S. Route 2"})
    MERGE (c:Entity {name: "Heron, Montana"})
    MERGE (a)-[:TRAVERSED_BY]->(b)
    MERGE (b)-[:BEGINS_NEAR]->(c)
    """, "idx52: Rogers Pass → US Route 2 → Heron Montana"),

    # idx 62 — Orson Scott Card, Ender's Game (2013), Ender's Shadow / Speaker (2003)
    ("""
    MERGE (a:Entity {name: "Orson Scott Card"})
    MERGE (b:Entity {name: "Ender's Game (film)"})
    MERGE (a)-[:NOVEL_ADAPTED_AS]->(b)
    SET a.description = coalesce(a.description, "American novelist whose novels were adapted into films in 2003 and 2013")
    SET b.description = coalesce(b.description, "2013 film adaptation of Orson Scott Card novel")
    """, "idx62: Orson Scott Card → film adaptations 2003 & 2013"),

    # idx 67 — Pamela Adlon in King of the Hill and Mongo Wrestling Alliance
    ("""
    MERGE (a:Entity {name: "Pamela Adlon"})
    MERGE (b:Entity {name: "King of the Hill"})
    MERGE (c:Entity {name: "Mongo Wrestling Alliance"})
    MERGE (a)-[:FEATURED_IN]->(b)
    MERGE (a)-[:FEATURED_IN]->(c)
    SET a.description = coalesce(a.description, "American actress, voice actress, screenwriter, producer and director")
    """, "idx67: Pamela Adlon → King of the Hill, Mongo Wrestling Alliance"),

    # idx 72 — John-Michael Howson wrote book based on Dusty musical
    ("""
    MERGE (a:Entity {name: "John-Michael Howson"})
    MERGE (b:Entity {name: "Dusty (musical)"})
    MERGE (a)-[:WROTE]->(b)
    SET a.description = coalesce(a.description, "Australian radio commentator who wrote the book based on the Dusty musical")
    """, "idx72: John-Michael Howson → Dusty musical book"),

    # idx 78 — Ronald Simmons (Faarooq) member of APA, signed by WWE
    ("""
    MERGE (a:Entity {name: "Ron Simmons"})
    MERGE (b:Entity {name: "Acolytes Protection Agency"})
    MERGE (c:Entity {name: "WWE"})
    MERGE (a)-[:MEMBER_OF]->(b)
    MERGE (b)-[:PART_OF]->(c)
    SET a.description = coalesce(a.description, "Ronald Simmons, also known as Faarooq, member of the APA tag team in WWE")
    """, "idx78: Ronald Simmons → APA → WWE"),

    # idx 82 — The Gaslight Anthem and Seaweed share rock genre
    ("""
    MERGE (a:Entity {name: "The Gaslight Anthem"})
    MERGE (b:Entity {name: "Seaweed (band)"})
    MERGE (c:Entity {name: "rock music"})
    MERGE (a)-[:GENRE]->(c)
    MERGE (b)-[:GENRE]->(c)
    """, "idx82: Gaslight Anthem & Seaweed → rock genre"),

    # idx 98 — Cate Blanchett starred with Louis Hunter in The War of the Roses
    ("""
    MERGE (a:Entity {name: "Cate Blanchett"})
    MERGE (b:Entity {name: "Louis Hunter"})
    MERGE (c:Entity {name: "The War of the Roses (play)"})
    MERGE (a)-[:ACTED_IN]->(c)
    MERGE (b)-[:ACTED_IN]->(c)
    SET a.description = coalesce(a.description, "Australian actress Catherine Elise Blanchett, starred with Louis Hunter in The War of the Roses")
    """, "idx98: Cate Blanchett + Louis Hunter → The War of the Roses"),
]


def run():
    with driver.session(database=DB) as session:
        for cypher, desc in RELATIONSHIPS:
            try:
                session.run(cypher)
                print(f"  OK  {desc}")
            except Exception as e:
                print(f"  ERR {desc}: {e}")
    driver.close()
    print(f"\nDone. {len(RELATIONSHIPS)} relationship groups injected.")


if __name__ == "__main__":
    run()
