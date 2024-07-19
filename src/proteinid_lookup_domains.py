# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:54:00 2024

@author: piercetf
"""

import requests

import asyncio
import aiohttp

import json

BASEURL = "https://rest.uniprot.org"

RUN = '/idmapping/run'

STATUS = '/idmapping/status'

RESULTS = '/idmapping/results'


SEARCH = '/uniprotkb/search'        
    
async def start_job(session :aiohttp.ClientSession, job_params :dict) -> dict:
    async with session.post(RUN, params=job_params) as resp:
        bodystr = await resp.text()
        data = json.loads(bodystr)
    return data

# async def check_job(session :aiohttp.ClientSession, jobid :str) -> bool:
#     async with session.get(STATUS + '/' + jobid, allow_redirects=False) as resp:
#         bodystr = await resp.text()
#         data = json.loads(bodystr)
#         finished = data['jobStatus'] == 'FINISHED'
#     return finished

# async def request_results(session :aiohttp.ClientSession, jobid :str):
#     async with session.get(RESULTS + '/' + jobid) as resp:
#         bodystr = await resp.text()
#         data = json.loads(bodystr)
#     return data

# async def get_results_when_ready(session :aiohttp.ClientSession, jobid :str):
#     ready = await check_job(session, jobid)
#     while not ready:
#         # only check once every 10 seconds to avoid hammering the service
#         await asyncio.sleep(10.0)
#         ready = await check_job(session, jobid)
#     results = await request_results(session, jobid)
#     return results

async def checkout_results(session :aiohttp.ClientSession, jobid :str):
    async with session.get(STATUS + '/' + jobid, allow_redirects=True) as resp:
        bodystr = await resp.text()
        data = json.loads(bodystr)
    return data


async def domain_search(session :aiohttp.ClientSession, idlist: list):
    domaintxt = {}
    for identity in idlist:
        params = {
            'query' : identity,
            'fields' : ['id', 
                        'xref_interpro',
                        'xref_brenda'
                        ]
            }
    
        async with session.get(SEARCH, params=params) as resp:
            bodystr = await resp.text()
            data = json.loads(bodystr)
            domaintxt[identity] = data
    return domaintxt


async def main():
    async with aiohttp.ClientSession(BASEURL) as session:
        # params = {
        #     'ids': '"P21802,P12345"',
        #     'from': 'UniProtKB_AC-ID',
        #     'to': 'UniRef90'
        #     }
        # jobdata = await start_job(session, params)
        # jobid = jobdata['jobId']
        # data = await checkout_results(session, jobid)
        ids = ['O43929', 'O15111', 'Q9BX70', 'Q9NPA0', 'Q14146', 
               'A0A087X0P0;Q02224', 'P17706', 'E9PMJ2;J3KP39;Q9BPY3', 'Q6NUQ1', 
               'G5E9V4;Q969F9', 'Q9GZV4', 'Q6Y288', 'A0A087WWY9;H0YJ83;Q8TB24', 
               'F8VRQ4;H0YHV1;Q96GM5', 'Q9H490', 'F5H538;J3KNB8;Q9Y6R4', 
               'B7ZBM3;Q8IVH2', 'Q92997', 'A0A087WZ68;Q68CX2;Q9BVN2', 
               'Q5TA50', 'Q16342', 'Q9BRU9']
        dbody = await domain_search(session, ids)
        return dbody


if __name__ == '__main__':
    asyncio.run(main())