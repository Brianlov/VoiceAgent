import asyncio
import os
import sys

sys.path.append(r'c:\Users\Brian ooi\Documents\code\VoiceAgentv1')
from Graph_service_purever2 import GraphRAGProcessor

async def test():
    p = GraphRAGProcessor('bolt://127.0.0.1:7687', 'neo4j', '20010808', 'hotpotqarag')
    res = await p.retrieve('Who directed a film written by the same pairing that later wrote Wild Wild West?', as_list=True)
    print(res)

asyncio.run(test())
