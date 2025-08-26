​​from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

from pydantic import Field
from typing import Any, Literal

import httpx
from urllib.parse import quote

import inspect

import os
import re
import sys

from concurrent.futures import ThreadPoolExecutor

from graph_tools.graph_contractor import ComplexGraphBuilder

VISION_API_BASE = os.getenv("VISION_API_BASE","http://localhost:5189/rest/tenants/default")
VISION_PAGE_SIZE = os.getenv("VISION_PAGE_SIZE","1000")
VISION_DISPLAY_PAGE_SIZE = os.getenv("VISION_DISPLAY_PAGE_SIZE", "20")  # Items per page for display
VISION_CODE = os.getenv("VISION_CODE",'False').lower() in ('true') 
VISION_VS = os.getenv("VISION_VS",'False').lower() in ('true') 
VISION_ROOT = os.getenv("VISION_ROOT","§{main_sources}§/")
MCP_PRINT = os.getenv("MCP_PRINT",'False').lower() in ('true') 

USER_AGENT = "vision-app/1.0"

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('mcp_img.log', maxBytes=10*1024*1024, backupCount=5),
        # logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

logger.info(f"VISION_API_BASE: {VISION_API_BASE}")
logger.info(f"VISION_ROOT: {VISION_ROOT}")
logger.info(f"VISION_CODE: {VISION_CODE}")
logger.info(f"VISION_VS: {VISION_VS}")

import json
import networkx as nx

def add_layout_positions(graph_json, layout_type='spring'):
    """
    Add layout positions to a graph defined in JSON format.
    
    Parameters:
    -----------
    graph_json : str or dict
        JSON string or dictionary containing nodes and links
    layout_type : str
        Type of layout algorithm to use ('spring', 'kamada_kawai', 'spectral', or 'shell')
        
    Returns:
    --------
    dict: Updated graph JSON with position information added to each node
    """
    # Parse JSON if it's a string
    if isinstance(graph_json, str):
        graph_data = json.loads(graph_json)
    else:
        graph_data = graph_json
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in graph_data["nodes"]:
        G.add_node(node["_id"], name=node["name"], nbObjects=node["nbObjects"])
    
    # Add edges
    for link in graph_data["links"]:
        G.add_edge(link["origin"]["_id"], link["target"]["_id"])
    
    # Compute layout positions
    if layout_type == 'spring':
        # Spring layout (force-directed)
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    elif layout_type == 'kamada_kawai':
        # Kamada-Kawai layout
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'spectral':
        # Spectral layout
        pos = nx.spectral_layout(G)
    elif layout_type == 'shell':
        # Shell layout
        pos = nx.shell_layout(G)
    else:
        # Default to spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Add position information to nodes
    for node in graph_data["nodes"]:
        node_id = node["_id"]
        if node_id in pos:
            node["x"] = float(pos[node_id][0])
            node["y"] = float(pos[node_id][1])
    
    return graph_data


def paginate_results(
    items: list,
    page: int = 1,
    page_size: int = None
) -> tuple[list, dict]:
    """
    Paginate a list of items client-side.
    
    Returns:
        tuple: (paginated_items, pagination_info)
    """
    if page_size is None:
        page_size = int(VISION_DISPLAY_PAGE_SIZE)
    
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size  # Ceiling division
    
    # Validate page number
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    
    paginated_items = items[start_idx:end_idx] if items else []
    
    pagination_info = {
        "current_page": page,
        "total_pages": total_pages,
        "page_size": page_size,
        "total_items": total_items,
        "showing_from": start_idx + 1 if paginated_items else 0,
        "showing_to": end_idx,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }
    
    return paginated_items, pagination_info


def format_pagination_info(pagination_info: dict) -> str:
    """Format pagination information for display."""
    if pagination_info["total_items"] == 0:
        return "No items to display."
    
    info = f"Showing items {pagination_info['showing_from']}-{pagination_info['showing_to']} of {pagination_info['total_items']} total"
    
    # Show page count when there are multiple pages
    if pagination_info["total_pages"] > 1:
        info = f"Page {pagination_info['current_page']} of {pagination_info['total_pages']} | " + info
    
    return info

def format_paginated_result(base_message: str, paginated_items_str: list[str], pagination_info: dict, extra_info: list[str] = None) -> str:
    """
    Formats the final result string including base message, pagination info,
    optional extra info, navigation hints, and paginated items.
    """
    result_parts = [base_message]

    # Add pagination info
    result_parts.append(format_pagination_info(pagination_info))

    # Add optional extra info (e.g., search criteria)
    if extra_info:
        result_parts.extend(extra_info)

    # Add navigation hints if there are multiple pages
    if pagination_info["total_pages"] > 1:
        nav_hints = []
        if pagination_info["has_previous"]:
            nav_hints.append(f"Previous: page={pagination_info['current_page']-1}")
        if pagination_info["has_next"]:
            nav_hints.append(f"Next: page={pagination_info['current_page']+1}")
        if nav_hints:
            result_parts.append(" | ".join(nav_hints))

    # Add formatted items
    if paginated_items_str:
        result_parts.append("") # Empty line before items
        result_parts.append("\n---\n".join(paginated_items_str))
    else:
         result_parts.append("No items to display on this page.")


    return "\n".join(result_parts)



class VisionServer:
    def __init__(self, base_url: str = VISION_API_BASE, 
                 user_agent: str = USER_AGENT, 
                 use_code: bool = VISION_CODE, 
                 use_vs: bool = VISION_VS,
                 page_size: str = VISION_PAGE_SIZE,
                 root_dir: str = VISION_ROOT):
        self.base_url = re.sub('(?<!^http:)(?<!^https:)//', '/', base_url)
        self.user_agent = user_agent
        self.root_dir = root_dir
        self.page_size = page_size
        self.use_code = use_code
        self.use_vs = use_vs

    def _get_base_url(self):
        return self.base_url

    def _get_root_dir(self):
        return self.root_dir

    def _get_use_code(self):
        return self.use_code

    def _get_use_vs(self):
        return self.use_vs

    async def _make_vision_request(self, url_query: str, accept: str = "application/json", operation: Literal["get","post"] = "get", body: str = None) -> dict[str, Any] | None:
        """Make a request to the Vision API with proper error handling."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept
        }
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/{url_query}"
                url = re.sub('(?<!^http:)(?<!^https:)//', '/', url)
                if operation == "post" and body is not None:
                    response = await client.post(url, headers=headers, data=body, timeout=30.0)
                else:
                    response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                return response.json() if accept == "application/json" else response.text
            except Exception:
                logger.error(f"url: {url}")
                logger.error(f"headers: {headers}")
                return None


def generic_format_function(data: dict) -> str:
    """Generic format function (TODO: adapt to different cases)"""
    formated_str = ''
    for k, v in data.items(): 
        formated_str += "{}: [{}]\n".format(k, v)
        
    return formated_str

        
mcp = FastMCP("vision", 
              dependencies=["mcp[cli]"]
        )

vision_server = VisionServer(VISION_API_BASE,USER_AGENT,VISION_CODE,VISION_PAGE_SIZE)

# PORTFOLIO WIDE TOOLS ####

@mcp.tool()
async def applications(ctx: Context,
    page: int = 1) -> str:
    """Available applications with Vision server. Paginated results.
    
    Args:
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications"
    await ctx.report_progress(progress)
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    # Apply pagination
    paginated_apps, pagination_info = paginate_results(applications, page)
    
    application_str = "\n---\n".join([generic_format_function(a) for a in paginated_apps])
    
    # result = f"Available applications:\n{format_pagination_info(pagination_info)}\n"
    
    # # Add navigation hints if there are multiple pages
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{application_str}"
    # return result

    return format_paginated_result(
        base_message="Available applications:",
        paginated_items_str=application_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def applications_transactions(ctx: Context,
                                    filters: str = None,
    page: int = 1) -> str:
    """Transactions from all applications in Vision. Paginated results.
       Filter transactions using query string format: "field:operator:value,field:operator:value"

       Available fields for "contains" operator:
       name, fullname, type

       Examples:
       - Simple filter: "name:contains:myapp"
       - Multiple filters: "name:contains:web,type:contains:post"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress =1

    criteria = {}
    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k in ['name','fullname','type'] and o == 'contains':
                    criteria[k] = v

    url_query = f"applications"
    await ctx.report_progress(progress)
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/transactions"
        first_arg = True
        for k, v in criteria.items():
             url_query += f"{"?" if first_arg else "&"}{k}={quote(v)}" # Use quote for filter values
             first_arg = False

        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            transactions = {
                "nb_transactions": len(data),
                "transactions": data,
                "criteria": criteria
            }
        else:
            transactions = {
                "nb_transactions": 0,
                "criteria": criteria
            }

        application_details.append({
            **info,
            **transactions
        })

    # Apply pagination
    paginated_details, pagination_info = paginate_results(application_details, page)
    
    application_str = "\n---\n".join(
        [generic_format_function(d) for d in paginated_details]
    )
    
    # result = f"Applications' transactions:\n{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{application_str}"
    # return result

    return format_paginated_result(
        base_message="Applications' transactions:",
        paginated_items_str=application_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def applications_data_graphs(ctx: Context,
                                   filters: str = None,
    page: int = 1) -> str:
    """Data entity interaction networks, a.k.a. data graphs, from all applications in Vision. Paginated results.
       Filter data graphs using query string format: "field:operator:value,field:operator:value"

       Available fields for "contains" operator:
       name, fullname, type

       Examples:
       - Simple filter: "name:contains:myapp"
       - Multiple filters: "name:contains:web,type:contains:post"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress =1

    criteria = {}
    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k in ['name','fullname','type'] and o == 'contains':
                    criteria[k] = v

    url_query = f"applications"
    await ctx.report_progress(progress)
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/datagraphs"
        first_arg = True
        for k, v in criteria.items():
             url_query += f"{"?" if first_arg else "&"}{k}={quote(v)}" # Use quote for filter values
             first_arg = False
 
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            datagraphs = {
                "nb_data_graphs": len(data),
                "data_graphs": data
            }
        else:
            datagraphs = {
                "nb_data_graphs": 0
            }

        application_details.append({
            **info,
            **datagraphs
        })


    # Apply pagination
    paginated_details, pagination_info = paginate_results(application_details, page)
    
    application_str = "\n---\n".join(
        [generic_format_function(d) for d in paginated_details]
    )
    
    # result = f"Applications' data entity interaction networks (a.k.a. data graphs):\n{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{application_str}"
    # return result

    return format_paginated_result(
        base_message="Applications' data entity interaction networks (a.k.a. data graphs):",
        paginated_items_str=application_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def applications_dependencies(ctx: Context,
    page: int = 1) -> str:
    """Inter-dependencies for all applications in Vision. Paginated results.
    
    Args:
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress =1

    url_query = f"applications"
    await ctx.report_progress(progress)
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/x-in-apps"
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            incoming_dependencies = {
                "nb_incoming_dependencies": len(data),
                "incoming_dependencies": data
            }
        else:
            incoming_dependencies = {
                "nb_incoming_dependencies": 0
            }

        url_query = f"applications/{application}/x-out-apps"
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            outgoing_dependencies = {
                "nb_outgoing_dependencies": len(data),
                "outgoing_dependencies": data
            }
        else:
            outgoing_dependencies = {
                "nb_outgoing_dependencies": 0
            }

        application_details.append({
            **info,
            **incoming_dependencies,
            **outgoing_dependencies
        })


    # Apply pagination
    paginated_details, pagination_info = paginate_results(application_details, page)
    
    application_str = "\n---\n".join(
        [generic_format_function(d) for d in paginated_details]
    )
    
    # result = f"Applications' inter-dependencies:\n{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{application_str}"
    # return result

    return format_paginated_result(
        base_message="Applications' inter-dependencies:",
        paginated_items_str=application_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def applications_quality_insights(ctx:Context,
    page: int = 1) -> str:
    """Quality insights for all applications in Vision. Paginated results.
    
    Args:
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications"
    await ctx.report_progress(progress)
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/insights/cve"
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            cves = {
                "nb_cves": len(data),
                "cves": data
            }
        else:
            cves = {
                "comment": "No CVE insights found for this application. The scanning for such issues was likely not configured."
            }

        url_query = f"applications/{application}/insights/cloud-detection-patterns"
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            cloud = {
                "nb_cloud_detection_patterns": len(data),
                "cloud_detection_patterns": data
            }
        else:
            cloud = {
                "comment": "No Cloud Maturity insights found for this application. The scanning for such issues was likely not configured."
            }

        url_query = f"applications/{application}/insights/green-detection-patterns"
        progress +=1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            green = {
                "nb_green_detection_patterns": len(data),
                "green_detection_patterns": data
            }
        else:
            green = {
                "comment": "No Green Deficiency insights found for this application. The scanning for such issues was likely not configured."
            }

        application_details.append({
            **info,
            **cves,
            **cloud,
            **green
        })


    # Apply pagination
    paginated_details, pagination_info = paginate_results(application_details, page)
    
    application_str = "\n---\n".join(
        [generic_format_function(d) for d in paginated_details]
    )
    
    # result = f"Applications' quality insights:\n{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{application_str}"
    # return result

    return format_paginated_result(
        base_message="Applications' quality insights:",
        paginated_items_str=application_str,
        pagination_info=pagination_info
    )


# APPLICATION WIDE TOOLS ####

@mcp.tool()
async def stats(application: str, ctx: Context,
    page: int = 1) -> str:
    """Get basic stats for an application. Paginated results.

    Args:
        application: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications/{application}/details"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    data = await vision_server._make_vision_request(url_query)

    if not data:
        return f"Unable to fetch stats of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(data) == 0:
        return "No stats for this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_data, pagination_info = paginate_results(data, page)
    
    data_str = [generic_format_function(d) for d in paginated_data]
    
    # result = f"Stats for {application}:\n{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(data_str)
    # return result

    return format_paginated_result(
        base_message=f"Stats for {application}:",
        paginated_items_str=data_str,
        pagination_info=pagination_info
    )

# @mcp.tool()
# async def advisors(application: str, ctx: Context) -> str:
#     """List modernization advisors for an application.

#     Args:
#         application: string
#     """
#     progress = 1

#     url_query = f"applications/{application}/advisors"
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#     await ctx.report_progress(progress)
#     data = await vision_server._make_vision_request(url_query)

#     if not data:
#         return f"Unable to fetch advisors of application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     if len(data) == 0:
#         return "No advisors for this application."

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     data_str = [generic_format_function(d) for d in data]
#     return "\n---\n".join(data_str) 

# @mcp.tool()
# async def advisor_rule_occurrences(application: str, ctx: Context,  id: str) -> str:
#     """Get occurrences of a rule supportring modernization advisors for an application identified by ID.

#     Args:
#         application: string
#         id: string
#     """
#     progress = 1

#     if type(id) is int:
#         id = str(id)

#     url_query = f"applications/{application}/advisors/rules/{id}/occurrences?limit={vision_server.page_size}"
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#     await ctx.report_progress(progress)
#     data = await vision_server._make_vision_request(url_query)

#     if not data:
#         return f"Unable to fetch advisors of application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     if len(data) == 0:
#         return "No advisors for this application."

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     data_str = [generic_format_function(d) for d in data]
#     return "\n---\n".join(data_str) 

@mcp.tool()
async def architectural_graph(application: str, ctx: Context, 
                              mode: Literal["nodes","links"] = "nodes", 
                              level: Literal["layer","component","sub-component","technology-category","element-type"] = "component",
                              layout: Literal["spring","kamada_kawai","spectral","shell","none"] = "none",
    page: int = 1) -> str:
    """Get architectural graph of an application, at a specific level. 
    * Layer (e.g., User Interaction)
    * Component (e.g., Web Interaction)
    * Sub-component (e.g., Web Communication)
    * Technology category (e.g., Javascript Communication)
    * Specific element type (e.g., Javascript Web Service Call)
    If layout is not 'none', add positions. Paginated results.

    Args:
        application: string
        mode: Literal["nodes","links"]
        level: Literal["layer","component","sub-component","technology-category","element-type"]
        layout: Literal["spring","kamada_kawai","spectral","shell","none]
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    level_idx = ["layer","component","sub-component","technology-category","element-type"].index(level)

    url_query = f"applications/{application}/levels/{str(level_idx+1)}/graph"
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch application {application} architectural graph with [{vision_server._get_base_url()}/{url_query}]."

    if layout != "none":
        graph = add_layout_positions(graph, layout_type=layout)

    items = graph[mode]

    if len(items) == 0:
        return f"No {mode} in architectural graph of application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Architectural graph {mode} for {application} at {level} level:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Architectural graph {mode} for {application} at {level} level:",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def architectural_graph_focus(application: str, 
                                    ctx: Context, area: str, 
                                    mode: Literal["nodes","links"] = "nodes", 
                                    level: Literal["layer","component","sub-component","technology-category","element-type"] = "component",
    page: int = 1) -> str:
    """Get architectural graph of an application, 
    at a specific level, 
    * Layer (e.g., User Interaction)
    * Component (e.g., Web Interaction)
    * Sub-component (e.g., Web Communication)
    * Technology category (e.g., Javascript Communication)
    * Specific element type (e.g., Javascript Web Service Call)
    focused on a specific area (e.g., communication). Paginated results.

    Args:
        application: string
        mode: Literal["nodes","links"]
        level: Literal["layer","component","sub-component","technology-category","element-type"]
        area: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    level_idx = ["layer","component","sub-component","technology-category","element-type"].index(level)

    url_query = f"applications/{application}/levels/{str(level_idx)}/graph"
    if area != "":
        url_query += f"?fullname={area}"
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch application {application} architectural graph with [{vision_server._get_base_url()}/{url_query}]."

    items = graph[mode]

    if len(items) == 0:
        return f"No {mode} in architectural graph of application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Architectural graph {mode} for {application} at {level} level (focused on {area}):\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Architectural graph {mode} for {application} at {level} level (focused on {area}):",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def quality_insights(application: str, 
                           ctx: Context,  
                           nature: Literal["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"],
    page: int = 1) -> str:
    """Get quality-related insights (CVE, cloud blockers, green deficiencies, structural flaws, ...) for an application. Paginated results.

    Args:
        application: string
        nature: Literal["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"]
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if nature not in ["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"]:
        return """{nature} is not a supported kind of insights. Only ["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"] are supported as of now."""

    # http://localhost:5189/rest/tenants/default/applications/FNT/insights/structural-flaws/detection-patterns/1020004/occurrences?limit=10

    if nature == "structural-flaws":
        url_query = f"applications/{application}/insights/{nature}/detection-patterns"
    else:
        url_query = f"applications/{application}/insights/{nature}"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    insights = await vision_server._make_vision_request(url_query)

    if not insights:
        return f"Unable to fetch insights of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(insights) == 0:
        if nature == "structural-flaws":
            return "No insights in this application."
        else:
            return f"No insights of kind {nature} in this application. The scanning for such issues was likely not configured."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_insights, pagination_info = paginate_results(insights, page)
    
    insights_str = [generic_format_function(i) for i in paginated_insights]
    
    # result = f"Quality insights ({nature}) for {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(insights_str)
    # return result

    return format_paginated_result(
        base_message=f"Quality insights ({nature}) for {application}:",
        paginated_items_str=insights_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def quality_insight_occurrences(application: str, 
                                      ctx: Context,  
                                      id: str, 
                                      nature: Literal["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"],
    page: int = 1) -> str:
    """Get occurrences of targeted quality-related insights (CVE, cloud blockers, green deficiencies, ...) for an application. Paginated results.

    Args:
        application: string
        nature: Literal["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"]
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    if nature not in ["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"]:
        return """{nature} is not a supported kind of insights. Only ["cloud-detection-patterns","green-detection-patterns","cve","structural-flaws"] are supported as of now."""

    if nature == "structural-flaws":
        url_query = f"applications/{application}/insights/{nature}/detection-patterns/{id}/occurrences?limit={vision_server.page_size}"
    else:
        url_query = f"applications/{application}/insights/{nature}/{id}/findings?limit={vision_server.page_size}"

    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    findings = await vision_server._make_vision_request(url_query)

    if not findings:
        return f"Unable to fetch insight occurrences of {nature} {id} application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(findings) == 0:
        return "No occurendes for insight in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_findings, pagination_info = paginate_results(findings, page)
    
    findings_str = [generic_format_function(f) for f in paginated_findings]
    
    # result = f"Occurrences of {nature} ({id}) in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(findings_str)
    # return result

    return format_paginated_result(
        base_message=f"Occurrences of {nature} ({id}) in {application}:",
        paginated_items_str=findings_str,
        pagination_info=pagination_info
    )

if vision_server._get_use_code(): 
    @mcp.tool()
    async def documents(application: str, 
                        ctx: Context, 
                        text: str, 
                        mode: Literal["element","graph","all"] = "all",
    page: int = 1) -> str:
        """Find documents about individual and/or graph of code elements in application matching ALL words in text argument. Paginated results.

        Args:
            application: string
            text: string
            mode: Literal["element","graph","all"]
            page: int (default: 1)
        """
        logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
        progress = 1

        if vision_server._get_use_vs():
            return "This tool is disabled when using the Vision Vector Store. Please use the get_usefull_documents from mcp-vector-store-vision instead."

        if mode in ["all","element"]:
            element_stat_url_query = f"applications/{application}/documents/stats"
            ctx.info(f"Processing {element_stat_url_query} for {ctx.request_id} of {ctx.client_id}")
            await ctx.report_progress(progress)
            element_stats = await vision_server._make_vision_request(element_stat_url_query)

        if mode in ["all","graph"]:
            graph_stat_url_query = f"applications/{application}/summaries/stats"
            ctx.info(f"Processing {graph_stat_url_query} for {ctx.request_id} of {ctx.client_id}")
            await ctx.report_progress(progress)
            graph_stats = await vision_server._make_vision_request(graph_stat_url_query)

        if not element_stats and not graph_stats:
            return f"Unable to fetch any documents for application {application} with [{vision_server._get_base_url()}/{element_stat_url_query}] and [{vision_server._get_base_url()}/{element_stat_url_query}]."

        if len(element_stats) == 0 and len(graph_stats) == 0:
            return f"No documents available at all for application {application}."

        docs = []
        if mode in ["all","element"]:
            element_url_query = f"applications/{application}/documents?limit={vision_server.page_size}"
            if text and text != '':
                element_url_query += f"&text={text}"
            ctx.info(f"Processing {element_url_query} for {ctx.request_id} of {ctx.client_id}")
            await ctx.report_progress(progress)
            docs = await vision_server._make_vision_request(element_url_query)

        summaries = []
        if mode in ["all","graph"]:
            graph_url_query = f"applications/{application}/summaries?limit={vision_server.page_size}"
            if text and text != '':
                graph_url_query += f"&text={text}"
            ctx.info(f"Processing {graph_url_query} for {ctx.request_id} of {ctx.client_id}")
            progress += 1
            await ctx.report_progress(progress)
            summaries = await vision_server._make_vision_request(graph_url_query)

        if not docs and not summaries:
            return f"Unable to fetch documents for application {application} with [{vision_server._get_base_url()}/{element_url_query}] and [{vision_server._get_base_url()}/{graph_url_query}]."

        if len(docs) == 0 and len(summaries) == 0:
            return f"No documents matching {text} for application {application}."
        
        ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)

        all_texts_str = []
        if len(docs) == 0:
            all_texts_str = [generic_format_function(s) for s in summaries]
        elif len(summaries) == 0:
            all_texts_str = [generic_format_function(d) for d in docs]
        else:
            all_texts_str = [generic_format_function(s) for s in summaries] + [generic_format_function(d) for d in docs]

        # Apply pagination
        paginated_texts, pagination_info = paginate_results(all_texts, page)
        
        all_texts_str = [generic_format_function(t) for t in paginated_texts]
        
        # result = f"Documents for {application} matching '{text}':\n"
        # result += f"{format_pagination_info(pagination_info)}\n"
        
        # if pagination_info["total_pages"] > 1:
        #     nav_hints = []
        #     if pagination_info["has_previous"]:
        #         nav_hints.append(f"Previous: page={page-1}")
        #     if pagination_info["has_next"]:
        #         nav_hints.append(f"Next: page={page+1}")
        #     result += " | ".join(nav_hints) + "\n"
        
        # result += "\n" + "\n---\n".join(all_texts_str)
        # return result

        return format_paginated_result(
            base_message=f"Documents for {application} matching '{text}':",
            paginated_items_str=all_texts_str,
            pagination_info=pagination_info
        )

@mcp.tool()
async def inter_applications_dependencies(application: str, ctx: Context) -> str:
    """Get inward and outward inter-application dependencies for application. Paginated results.

    Args:
        application: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    x_in_url_query = f"applications/{application}/x-in-apps"
    ctx.info(f"Processing {x_in_url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    x_in_apps = await vision_server._make_vision_request(x_in_url_query)

    x_out_url_query = f"applications/{application}/x-out-apps"
    ctx.info(f"Processing {x_out_url_query} for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)
    x_out_apps = await vision_server._make_vision_request(x_out_url_query)

    if not x_in_apps and not x_out_apps:
        return f"Unable to fetch inward inter-application dependencies for application {application} with [{vision_server._get_base_url()}/{x_in_url_query}] and  [{vision_server._get_base_url()}/{x_out_url_query}]."

    if len(x_in_apps) == 0 and len(x_out_apps) == 0:
        return "No inter-application dependencies for this application."
    elif len(x_in_apps) == 0:
        x_apps = x_out_apps
    elif len(x_out_apps) == 0:
        x_apps = x_in_apps
    else:
        x_apps = x_in_apps.extend(x_out_apps)

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)


    # Apply pagination
    paginated_apps, pagination_info = paginate_results(x_apps, page)
    
    x_apps_str = [generic_format_function(t) for t in paginated_apps]
    
    # result = f"Inter-application dependencies for {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(x_apps_str)
    # return result

    return format_paginated_result(
        base_message=f"Inter-application dependencies for {application}:",
        paginated_items_str=x_apps_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def inter_app_detailed_dependencies(application: str, 
                                          other_app: str, 
                                          ctx: Context,
    page: int = 1) -> str:
    """Get inward and outward inter-application detailed dependencies between applications. Paginated results.

    Args:
        application: string
        other_application: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    x_in_url_query = f"applications/{application}/x-in-apps/{other_application}"
    ctx.info(f"Processing {x_in_url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    x_in_apps = await vision_server._make_vision_request(x_in_url_query)

    x_out_url_query = f"applications/{application}/x-out-apps/{other_application}"
    ctx.info(f"Processing {x_out_url_query} for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)
    x_out_apps = await vision_server._make_vision_request(x_out_url_query)

    if not x_in_apps and not x_out_apps:
        return f"Unable to fetch inward inter-application detailed dependencies between application {application} and application {other_application} with [{vision_server._get_base_url()}/{x_in_url_query}] and [{vision_server._get_base_url()}/{x_out_url_query}]."

    if len(x_in_apps) == 0 and len(x_out_apps) == 0:
        return f"No inter-application detailed dependencies between application {application} and application {other_application}."
    elif len(x_in_apps) == 0:
        x_apps = x_out_apps
    elif len(x_out_apps) == 0:
        x_apps = x_in_apps
    else:
        x_apps = x_in_apps.extend(x_out_apps)

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_apps, pagination_info = paginate_results(x_apps, page)
    
    x_apps_str = [generic_format_function(t) for t in paginated_apps]
    
    # result = f"Detailed dependencies between {application} and {other_app}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(x_apps_str)
    # return result

    return format_paginated_result(
        base_message=f"Detailed dependencies between {application} and {other_app}:",
        paginated_items_str=x_apps_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def packages(application: str, 
                   ctx: Context,
    page: int = 1) -> str:
    """Get packages for an application. Paginated results.

    Args:
        application: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications/{application}/packages?limit={vision_server.page_size}"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    packages = await vision_server._make_vision_request(url_query)

    if not packages:
        return f"Unable to fetch packages of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(packages) == 0:
        return "No packages in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_packages, pagination_info = paginate_results(packages, page)
    
    packages_str = [generic_format_function(t) for t in paginated_packages]
    
    # result = f"Packages for {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(packages_str)
    # return result

    return format_paginated_result(
        base_message=f"Packages for {application}:",
        paginated_items_str=packages_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def package_interactions(application: str, 
                               component: str, 
                               version: str, 
                               ctx: Context,
    page: int = 1) -> str:
    """Get interactions with packages for an application. Paginated results.

    Args:
        application: string
        component: string
        version: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications/{application}/packages/interactions?component={component}&version={version}"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    package_interactions = await vision_server._make_vision_request(url_query)

    if not package_interactions:
        return f"Unable to fetch interactions for package {component} (version {version}) and application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(package_interactions) == 0:
        return "No interactions with package in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_interactions, pagination_info = paginate_results(package_interactions, page)
    
    package_interactions_str = [
        generic_format_function(t) for t in paginated_interactions
    ]
    
    # result = f"Interactions with package {component} (v{version}) in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(package_interactions_str)
    # return result

    return format_paginated_result(
        base_message=f"Interactions with package {component} (v{version}) in {application}:",
        paginated_items_str=package_interactions_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def transactions(application: str, ctx: Context,
                       filters: str = None,
    page: int = 1) -> str:
    """Get transactions for an application. Paginated results.
       Filter transactions using query string format: "field:operator:value,field:operator:value"

       Available fields for "contains" operator:
       name, fullname, type

       Examples:
       - Simple filter: "name:contains:myapp"
       - Multiple filters: "name:contains:web,type:contains:post"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications/{application}/transactions?limit={vision_server.page_size}"

    criteria = {}
    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k in ['name','fullname','type'] and o == 'contains':
                    url_query += f"&{k}={quote(v)}"
                    criteria[k] = v


    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    transactions = await vision_server._make_vision_request(url_query)

    if not transactions:
        return f"Unable to fetch transactions of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(transactions) == 0:
        return "No transactions in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_transactions, pagination_info = paginate_results(transactions, page)
    
    transactions_str = [generic_format_function(t) for t in paginated_transactions]
    
    # result = f"Transactions for {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(transactions_str)
    # return result
    # Build extra info list for criteria
    criteria_info = []
    if criteria:
        criteria_info.append(f"Search criteria: {', '.join([f'{k}={v}' for k, v in criteria.items()])}")

    return format_paginated_result(
        base_message=f"Transactions for {application}:",
        paginated_items_str=transactions_str,
        pagination_info=pagination_info,
        extra_info=criteria_info
    )


@mcp.tool()
async def transaction_objects_with_insights(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get objects with insights in a transaction identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/transactions/{id}/nodes_with_insights"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    summary = await vision_server._make_vision_request(url_query)

    if not summary:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No objects with insights in transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_summary, pagination_info = paginate_results(summary, page)
    
    summary_str = [generic_format_function(s) for s in paginated_summary]
    
    # result = f"Objects with insights in transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(summary_str)
    # return result

    return format_paginated_result(
        base_message=f"Objects with insights in transaction {id} in {application}:",
        paginated_items_str=summary_str,
        pagination_info=pagination_info
    )



@mcp.tool()
async def transaction_complex_objects(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get complex objects in a transaction identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/transactions/{id}/complex_nodes"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    summary = await vision_server._make_vision_request(url_query)

    if not summary:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No complex objects in transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_summary, pagination_info = paginate_results(summary, page)
    
    summary_str = [generic_format_function(s) for s in paginated_summary]
    
    # result = f"Complex objects in transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(summary_str)
    # return result

    return format_paginated_result(
        base_message=f"Complex objects in transaction {id} in {application}:",
        paginated_items_str=summary_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def transaction_documents(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get documents for a transaction identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/transactions/{id}/summary"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    summary = await vision_server._make_vision_request(url_query)

    if not summary:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No document for transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_summary, pagination_info = paginate_results(summary, page)
    
    summary_str = [generic_format_function(s) for s in paginated_summary]
    
    # result = f"Documents for transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(summary_str)
    # return result

    return format_paginated_result(
        base_message=f"Documents for transaction {id} in {application}:",
        paginated_items_str=summary_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def add_transaction_document(application: str, ctx: Context,  id: str, text: str) -> str:
    """Add a document to a transaction identified by ID in an application.

    Args:
        application: string
        id: string
        text: string
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/transactions/{id}/summary"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    
    print(f"Adding document to transaction {id} in application {application} with POST [{vision_server._get_base_url()}/{url_query}]")
    print(f"Document content: {text}")

    summary = await vision_server._make_vision_request(url_query, operation="post", body=text)

    if not summary:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No document added to transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    summary_str = [generic_format_function(s) for s in summary]
    return "\n---\n".join(summary_str)

@mcp.tool()
async def transaction_graph(application: str, 
                            ctx: Context,  
                            id: str, 
                            mode: Literal["nodes","links"] = "nodes", 
                            granularity: Literal["types","elements"] = "types",
    page: int = 1) -> str:
    """Get nodes or links at selected level of granularity of a transaction identified by ID in an application. Paginated results.
    Tips: for large graphs, use "types" granularity and/or "transaction_graph_focus" tool to focus on most complex objects first. 
    "transaction_objects_with_insights" and "transaction_complex_objects" tools are also an option to get started.

    Args:
        application: string
        id: string
        mode: Literal["nodes","links"]
        granularity: Literal["types","elements"]
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    # if mode == "type-level":
    #     url_query = f"applications/{application}/transactions/{id}/type-level-graph"
    # else:
    url_query = f"applications/{application}/transactions/{id}/graph"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."


    if granularity == "types":
        nodes = graph["nodes"]
        links = graph["links"]
        nodes_dict = {i["id"]: i for i in nodes}

        types = {}
        type_links = {}

        for i in nodes:
            progress += 1
            await ctx.report_progress(progress)

            elt_type = i["type"]
            if elt_type in types.keys():
                type_info = types[elt_type]
            else:
                type_info = {
                    "type": elt_type,
                    "count": 0,
                    "nb_start_points": 0,
                    "nb_end_points": 0
                }
            type_info["count"] += 1
            types[elt_type] = type_info
        
        # types = [v for v in types.values()]

        for i in links:
            progress += 1
            await ctx.report_progress(progress)

            origin = i["origin"]
            target = i["target"]

            origin_type = nodes_dict[origin["id"]]["type"]
            origin_is_start = origin["isStartPoint"]
            
            target_type = nodes_dict[target["id"]]["type"]
            target_is_end = target["isEndPoint"]
            
            link_type = i["type"]
            depth = i["depth"]

            type_link_key = f"{origin_type} -> {target_type}"
            if type_link_key in type_links.keys():
                type_link_info = type_links[type_link_key]
            else:       
                type_link_info = {
                    "origin": origin_type,
                    "target": target_type,
                    "link_type": link_type,
                    "count": 0,
                    "depth_range": {
                        "min": 0,
                        "max": 0
                    }
                }
            type_link_info["count"] += 1
            if depth < type_link_info["depth_range"]["min"]:
                type_link_info["depth_range"]["min"] = depth
            if depth > type_link_info["depth_range"]["max"]:
                type_link_info["depth_range"]["max"] = depth

            if origin_is_start:
                types[origin_type]["nb_start_points"] += 1
            if target_is_end:
                types[target_type]["nb_end_points"] += 1

            type_links[type_link_key] = type_link_info

        # type_links = [v for v in type_links.values()]

        graph = {
            "nodes": types,
            "links": type_links
        }

    items = graph[mode]

    if len(items) == 0:
        return f"No {granularity} {mode} in transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    if isinstance(items, dict):
        items_list = list(items.values())
    else:
        items_list = items

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items_list, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Transaction graph {mode} ({granularity}) for transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Transaction graph {mode} ({granularity}) for transaction {id} in {application}:",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def transaction_graph_focus(application: str, 
                            ctx: Context,  
                            id: str, 
                            mode: Literal["nodes","links","sequencing"] = "nodes", 
                            focus_size: int = 50,
    page: int = 1) -> str:
    """Get nodes, links, or sequencing of a reduced graph of a transaction identified by ID in an application, focused on most complex objects. Paginated results.

    Args:
        application: string
        id: string
        mode: Literal["nodes","links","sequencing"]
        focus_size: int
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/transactions/{id}/graph"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    url_query = f"applications/{application}/transactions/{id}/complex_nodes"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)
    complex_nodes = await vision_server._make_vision_request(url_query)

    if not complex_nodes:
        return f"Unable to fetch complex objects of transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    builder = ComplexGraphBuilder(
        graph, 
        complex_nodes[0:focus_size]
    )
    focus_graph = builder.build_complex_graph()
    
    if mode == "sequencing":

        nodes = focus_graph["nodes"]
        links = focus_graph["links"]
        nodes_dict = {i["id"]: i for i in nodes}

        nodes_with_depth = nodes_dict
        
        for l in links:
            progress += 1
            await ctx.report_progress(progress)

            origin = l["origin"]
            target = l["target"]
            depth = l["depth"] 

            origin_id = origin["id"]
            target_id = target["id"]

            n_orig = nodes_with_depth[origin_id]
            if "processing_seq" not in n_orig:
                n_orig["processing_seq"] = -1
            n_target = nodes_with_depth[target_id]
            if "processing_seq" not in n_target:
                n_target["processing_seq"] = -1
            
            if "after" not in n_target:
                n_target["after"] = [origin_id]
            else:
                n_target["after"].append(origin_id)
            
            if depth > n_orig["processing_seq"]:
                n_orig["processing_seq"] = depth

            if (depth + 1) > n_target["processing_seq"]:
                n_target["processing_seq"] = depth + 1

            nodes_with_depth[origin_id] = n_orig
            nodes_with_depth[target_id] = n_target

        items = nodes_with_depth

        if len(items) == 0:
            return f"Unable to compute sequencing data for data entity interaction network (a.k.a. data graph) with ID {id} in application {application}."


    else:
        items = focus_graph[mode]

        if len(items) == 0:
            return f"No {mode} in transaction with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    if isinstance(items, dict):
        items_list = list(items.values())
    else:
        items_list = items

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items_list, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Focused transaction graph {mode} for transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Focused transaction graph {mode} for transaction {id} in {application}:",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )


# @mcp.tool()
# async def transaction_sequencing(application: str, 
#                                  ctx: Context,  
#                                  id: str,
#                                  size: int = 50,
#     page: int = 1) -> str:
#     """Get sequencing data for greedy summarization of transaction identified by ID in an application focused on most complex objects. 
#     Paginated results.

#     Args:
#         application: string
#         id: string
#         size: int
#         page: int (default: 1)
#     """
#     logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
#     progress = 1

#     if type(id) is int:
#         id = str(id)

#     # if mode == "type-level":
#     #     url_query = f"applications/{application}/transactions/{id}/type-level-graph"
#     # else:
#     url_query = f"applications/{application}/transactions/{id}/graph"
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    
#     await ctx.report_progress(progress)
    
#     graph = await vision_server._make_vision_request(url_query)

#     if not graph:
#         return f"Unable to fetch transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     url_query = f"applications/{application}/datagraphs/{id}/complex_nodes"
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    
#     progress += 1
#     await ctx.report_progress(progress)
    
#     complex_nodes = await vision_server._make_vision_request(url_query)

#     if not complex_nodes:
#         return f"Unable to fetch complex objects of transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     builder = ComplexGraphBuilder(
#         graph, 
#         complex_nodes[0:size]
#     )
#     focus_graph = builder.build_complex_graph()

#     nodes = focus_graph["nodes"]
#     links = focus_graph["links"]
#     nodes_dict = {i["id"]: i for i in nodes}

#     nodes_with_depth = nodes_dict
    
#     for l in links:
#         progress += 1
#         await ctx.report_progress(progress)

#         origin = l["origin"]
#         target = l["target"]
#         depth = l["depth"] 

#         origin_id = origin["id"]
#         target_id = target["id"]

#         n_orig = nodes_with_depth[origin_id]
#         if "processing_seq" not in n_orig:
#             n_orig["processing_seq"] = -1
#         n_target = nodes_with_depth[target_id]
#         if "processing_seq" not in n_target:
#             n_target["processing_seq"] = -1
        
#         if depth > n_orig["processing_seq"]:
#             n_orig["processing_seq"] = depth

#         if (depth + 1) > n_target["processing_seq"]:
#             n_target["processing_seq"] = depth + 1

#         nodes_with_depth[origin_id] = n_orig
#         nodes_with_depth[target_id] = n_target

#     graph = {
#         "nodes": nodes_with_depth,
#         "links": links
#     }

#     items = nodes_with_depth

#     if len(items) == 0:
#         return f"Unable to compute sequencing data for transaction with ID {id} in application {application}."

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     # Apply pagination
#     paginated_items, pagination_info = paginate_results(items, page)
    
#     items_str = [generic_format_function(i) for i in paginated_items]
    
#     # result = f"Sequencing data for transaction {id} in {application}:\n"
#     # result += f"{format_pagination_info(pagination_info)}\n"
    
#     # if pagination_info["total_pages"] > 1:
#     #     nav_hints = []
#     #     if pagination_info["has_previous"]:
#     #         nav_hints.append(f"Previous: page={page-1}")
#     #     if pagination_info["has_next"]:
#     #         nav_hints.append(f"Next: page={page+1}")
#     #     result += " | ".join(nav_hints) + "\n"
    
#     # result += "\n" + "\n---\n".join(items_str)
#     # return result

#     return format_paginated_result(
#         base_message=f"Sequencing data for transaction {id} in {application}:",
#         paginated_items_str=items_str,
#         pagination_info=pagination_info
#     )

@mcp.tool()
async def data_graphs(application: str, ctx: Context,
                      filters: str = None,
    page: int = 1) -> str:
    """Get data entity interaction networks (a.k.a. data graphs) for an application. Paginated results.
       Filter data graphs using query string format: "field:operator:value,field:operator:value"

       Available fields for "contains" operator:
       name, fullname, type

       Examples:
       - Simple filter: "name:contains:myapp"
       - Multiple filters: "name:contains:web,type:contains:post"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    url_query = f"applications/{application}/datagraphs?limit={vision_server.page_size}"
        
    criteria = {}
    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k in ['name','fullname','type'] and o == 'contains':
                    url_query += f"&{k}={quote(v)}"
                    criteria[k] = v

    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    datagraphs = await vision_server._make_vision_request(url_query)

    if not datagraphs:
        return f"Unable to fetch data entity interaction networks (a.k.a. data graphs) of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(datagraphs) == 0:
        return "No data entity interaction networks (a.k.a. data graphs) in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_datagraphs, pagination_info = paginate_results(datagraphs, page)
    
    datagraphs_str = [generic_format_function(t) for t in paginated_datagraphs]
    
    # result = f"Data graphs for {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(datagraphs_str)
    # return result
    # Build extra info list for criteria
    criteria_info = []
    if criteria:
        criteria_info.append(f"Search criteria: {', '.join([f'{k}={v}' for k, v in criteria.items()])}")

    return format_paginated_result(
        base_message=f"Data graphs for {application}:",
        paginated_items_str=datagraphs_str,
        pagination_info=pagination_info,
        extra_info=criteria_info
    )



@mcp.tool()
async def data_graph_objects_with_insights(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get objects with insights in a data interaction network, a.k.a., data graph, identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/datagraphs/{id}/nodes_with_insights"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    summary = await vision_server._make_vision_request(url_query)

    if not summary:
        return f"Unable to fetch data graph with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No objects with insights in data graph with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_summary, pagination_info = paginate_results(summary, page)
    
    summary_str = [generic_format_function(s) for s in paginated_summary]
    
    # result = f"Objects with insights in data graph {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(summary_str)
    # return result

    return format_paginated_result(
        base_message=f"Objects with insights in data graph {id} in {application}:",
        paginated_items_str=summary_str,
        pagination_info=pagination_info
    )



@mcp.tool()
async def data_graph_complex_objects(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get complex objects in a data interaction network, a.k.a., data graph, identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/datagraphs/{id}/complex_nodes"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    summary = await vision_server._make_vision_request(url_query)

    if not summary:
        return f"Unable to fetch data graph with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No complex objects in data graph with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_summary, pagination_info = paginate_results(summary, page)
    
    summary_str = [generic_format_function(s) for s in paginated_summary]
    
    # result = f"Complex objects in data graph {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(summary_str)
    # return result

    return format_paginated_result(
        base_message=f"Complex objects in data graph {id} in {application}:",
        paginated_items_str=summary_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def data_graph_documents(application: str, ctx: Context,  id: str,
    page: int = 1) -> str:
    """Get documents for a data entity interaction network (a.k.a. data graph) identified by ID in an application. Paginated results.

    Args:
        application: string
        id: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/datagraphs/{id}/document"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    document = await vision_server._make_vision_request(url_query)

    if not document:
        return f"Unable to fetch data entity interaction network (a.k.a. data graph) with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(document) == 0:
        return f"No documnet for data entity interaction network (a.k.a. data graph) with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_document, pagination_info = paginate_results(document, page)
    
    document_str = [generic_format_function(s) for s in paginated_document]
    
    # result = f"Documents for data graph {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(document_str)
    # return result

    return format_paginated_result(
        base_message=f"Documents for data graph {id} in {application}:",
        paginated_items_str=document_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def add_datagraph_document(application: str, ctx: Context,  id: str, text: str) -> str:
    """Add a document to a data interaction network (a.k.a. data graph) identified by ID in an application.

    Args:
        application: string
        id: string
        text: string
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/datagraphs/{id}/summary"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    
    print(f"Adding document to data graph {id} in application {application} with POST [{vision_server._get_base_url()}/{url_query}]")
    print(f"Document content: {text}")

    summary = await vision_server._make_vision_request(url_query, operation="post", body=text)

    if not summary:
        return f"Unable to fetch data graph with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No document added to data graph with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    summary_str = [generic_format_function(s) for s in summary]
    return "\n---\n".join(summary_str)

@mcp.tool()
async def data_graph_graph(application: str, 
                           ctx: Context,  
                           id: str, 
                           mode: Literal["nodes","links"] = "nodes", 
                           granularity: Literal["types","elements"] = "types",
    page: int = 1) -> str:
    """Get nodes or links at selected level of granularity of a transactiondata entity interaction network (a.k.a. data graph) identified by ID in an application. Paginated results.
    Tips: for large graphs, use "types" granularity and/or "data_graph_graph_focus" tool to focus on most complex objects first. 
    "data_graph_objects_with_insights" and "data_graph_complex_objects" tools are also an option to get started.
    
    Args:
        application: string
        id: string
        mode: Literal["nodes","links"]
        granularity: Literal["types","elements"]
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    # if mode == "type-level":
    #     url_query = f"applications/{application}/datagraphs/{id}/type-level-graph"
    # else:
    url_query = f"applications/{application}/datagraphs/{id}/graph"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch data entity interaction network (a.k.a. data graph) with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."


    if granularity == "types":
        nodes = graph["nodes"]
        links = graph["links"]
        nodes_dict = {i["id"]: i for i in nodes}

        types = {}
        type_links = {}

        for i in nodes:
            progress += 1
            await ctx.report_progress(progress)

            elt_type = i["type"]
            if elt_type in types.keys():
                type_info = types[elt_type]
            else:
                type_info = {
                    "type": elt_type,
                    "count": 0,
                    "nb_start_points": 0,
                    "nb_end_points": 0
                }
            type_info["count"] += 1
            types[elt_type] = type_info
        
        # types = [v for v in types.values()]

        for i in links:
            progress += 1
            await ctx.report_progress(progress)

            origin = i["origin"]
            target = i["target"]

            origin_type = nodes_dict[origin["id"]]["type"]
            origin_is_start = origin["isStartPoint"]
            
            target_type = nodes_dict[target["id"]]["type"]
            target_is_end = target["isEndPoint"]
            
            link_type = i["type"]
            depth = i["depth"]

            type_link_key = f"{origin_type} -> {target_type}"
            if type_link_key in type_links.keys():
                type_link_info = type_links[type_link_key]
            else:       
                type_link_info = {
                    "origin": origin_type,
                    "target": target_type,
                    "link_type": link_type,
                    "count": 0,
                    "depth_range": {
                        "min": 0,
                        "max": 0
                    }
                }
            type_link_info["count"] += 1
            if depth < type_link_info["depth_range"]["min"]:
                type_link_info["depth_range"]["min"] = depth
            if depth > type_link_info["depth_range"]["max"]:
                type_link_info["depth_range"]["max"] = depth

            if origin_is_start:
                types[origin_type]["nb_start_points"] += 1
            if target_is_end:
                types[target_type]["nb_end_points"] +=1

            type_links[type_link_key] = type_link_info

        # type_links = [v for v in type_links.values()]

        graph = {
            "nodes": types,
            "links": type_links
        }

    items = graph[mode]

    if len(items) == 0:
        return f"No {granularity} {mode} in data entity interaction network (a.k.a. data graph) with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    if isinstance(items, dict):
        items_list = list(items.values())
    else:
        items_list = items

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items_list, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Data graph {mode} ({granularity}) for data graph {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Data graph {mode} ({granularity}) for data graph {id} in {application}:",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def data_graph_graph_focus(application: str, 
                            ctx: Context,  
                            id: str, 
                            mode: Literal["nodes","links","sequencing"] = "nodes", 
                            focus_size: int = 50,
    page: int = 1) -> str:
    """Get nodes, links, or sequencing of a reduced graph of a data interaction network, a.k.a., data graph, identified by ID in an application, focused on most complex objects. Paginated results.

    Args:
        application: string
        id: string
        mode: Literal["nodes","links","sequencing"]
        focus_size: int
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/datagraphs/{id}/graph"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    graph = await vision_server._make_vision_request(url_query)

    if not graph or mode not in graph:
        return f"Unable to fetch data graph with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    url_query = f"applications/{application}/datagraphs/{id}/complex_nodes"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)
    complex_nodes = await vision_server._make_vision_request(url_query)

    if not complex_nodes:
        return f"Unable to fetch complex objects of data graph with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    builder = ComplexGraphBuilder(
        graph, 
        complex_nodes[0:focus_size]
    )
    focus_graph = builder.build_complex_graph()
    
    if mode == "sequencing":

        nodes = focus_graph["nodes"]
        links = focus_graph["links"]
        nodes_dict = {i["id"]: i for i in nodes}

        nodes_with_depth = nodes_dict
        
        for l in links:
            progress += 1
            await ctx.report_progress(progress)

            origin = l["origin"]
            target = l["target"]
            depth = l["depth"] 

            origin_id = origin["id"]
            target_id = target["id"]

            n_orig = nodes_with_depth[origin_id]
            if "processing_seq" not in n_orig:
                n_orig["processing_seq"] = -1
            n_target = nodes_with_depth[target_id]
            if "processing_seq" not in n_target:
                n_target["processing_seq"] = -1
            
            if "after" not in n_target:
                n_target["after"] = [origin_id]
            else:
                n_target["after"].append(origin_id)
            
            if depth > n_orig["processing_seq"]:
                n_orig["processing_seq"] = depth

            if (depth + 1) > n_target["processing_seq"]:
                n_target["processing_seq"] = depth + 1

            nodes_with_depth[origin_id] = n_orig
            nodes_with_depth[target_id] = n_target

        items = nodes_with_depth

        if len(items) == 0:
            return f"Unable to compute sequencing data for data entity interaction network (a.k.a. data graph) with ID {id} in application {application}."


    else:    
        items = focus_graph[mode]

        if len(items) == 0:
            return f"No {mode} in data graph with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    if isinstance(items, dict):
        items_list = list(items.values())
    else:
        items_list = items

    # Apply pagination
    paginated_items, pagination_info = paginate_results(items_list, page)
    
    items_str = [generic_format_function(i) for i in paginated_items]
    
    # result = f"Focused transaction graph {mode} for transaction {id} in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(items_str)
    # return result

    return format_paginated_result(
        base_message=f"Focused transaction graph {mode} for transaction {id} in {application}:",
        paginated_items_str=items_str,
        pagination_info=pagination_info
    )


# @mcp.tool()
# async def data_graph_sequencing(application: str, 
#                                 ctx: Context, 
#                                 id: str,
#                                 size: int = 50, 
#     page: int = 1) -> str:
#     """Get sequencing data for greedy summarization of a data interaction network (a.k.a. data graph) identified by ID in an application and focused on most complext objects. 
#     Paginated results.

#     Args:
#         application: string
#         id: string
#         zise,
#         page: int (default: 1)
#     """
#     logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
#     progress = 1

#     if type(id) is int:
#         id = str(id)

#     # if mode == "type-level":
#     #     url_query = f"applications/{application}/datagraphs/{id}/type-level-graph"
#     # else:
#     url_query = f"applications/{application}/datagraphs/{id}/graph"
    
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#     await ctx.report_progress(progress)
    
#     graph = await vision_server._make_vision_request(url_query)

#     if not graph:
#         return f"Unable to fetch data entity interaction network (a.k.a. data graph) with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     url_query = f"applications/{application}/datagraphs/{id}/complex_nodes"
#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    
#     progress += 1
#     await ctx.report_progress(progress)
    
#     complex_nodes = await vision_server._make_vision_request(url_query)

#     if not complex_nodes:
#         return f"Unable to fetch complex objects of transaction with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     builder = ComplexGraphBuilder(
#         graph, 
#         complex_nodes[0:size]
#     )
#     focus_graph = builder.build_complex_graph()

#     nodes = focus_graph["nodes"]
#     links = focus_graph["links"]
#     nodes_dict = {i["id"]: i for i in nodes}

#     nodes_with_depth = nodes_dict
    
#     for l in links:
#         progress += 1
#         await ctx.report_progress(progress)

#         origin = l["origin"]
#         target = l["target"]
#         depth = l["depth"] 

#         origin_id = origin["id"]
#         target_id = target["id"]

#         n_orig = nodes_with_depth[origin_id]
#         if "processing_seq" not in n_orig:
#             n_orig["processing_seq"] = -1
#         n_target = nodes_with_depth[target_id]
#         if "processing_seq" not in n_target:
#             n_target["processing_seq"] = -1
        
#         if "after" not in n_target:
#             n_target["after"] = [origin_id]
#         else:
#             n_target["after"].append(origin_id)
        
#         if depth > n_orig["processing_seq"]:
#             n_orig["processing_seq"] = depth

#         if (depth + 1) > n_target["processing_seq"]:
#             n_target["processing_seq"] = depth + 1

#         nodes_with_depth[origin_id] = n_orig
#         nodes_with_depth[target_id] = n_target

#     items = nodes_with_depth

#     if len(items) == 0:
#         return f"Unable to compute sequencing data for data entity interaction network (a.k.a. data graph) with ID {id} in application {application}."

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     # Apply pagination
#     paginated_items, pagination_info = paginate_results(items, page)
    
#     items_str = [generic_format_function(i) for i in paginated_items]
    
#     # result = f"Sequencing data for data graph {id} in {application}:\n"
#     # result += f"{format_pagination_info(pagination_info)}\n"
    
#     # if pagination_info["total_pages"] > 1:
#     #     nav_hints = []
#     #     if pagination_info["has_previous"]:
#     #         nav_hints.append(f"Previous: page={page-1}")
#     #     if pagination_info["has_next"]:
#     #         nav_hints.append(f"Next: page={page+1}")
#     #     result += " | ".join(nav_hints) + "\n"
    
#     # result += "\n" + "\n---\n".join(items_str)
#     # return result

#     return format_paginated_result(
#         base_message=f"Sequencing data for data graph {id} in {application}:",
#         paginated_items_str=items_str,
#         pagination_info=pagination_info
#     )


# OBJECTS TOOLS ####

# @mcp.tool()
# async def objects(application: str, 
#                   ctx: Context, 
#                   name: str = None, 
#                   qualified_name: str = None, 
#                   file_path: str = None, 
#                   type: str = None,
#     page: int = 1) -> str:
#     """Get objects in an application matching identification criteria. Paginated results.

#     Args:
#         application: string
#         name: string
#         qualified_name: string
#         file_path: string
#         type: string
#         page: int (default: 1)
#     """
#     progress = 1

#     url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
#     if name is not None:
#         url_query += f"&name={name}"

#     if qualified_name is not None:
#         url_query += f"&fullname={qualified_name}"

#     if file_path is not None:
#         url_query += f"&filepath={quote(file_path)}"

#     if type is not None:
#         url_query += f"&type={type}"

#     ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)
#     data = await vision_server._make_vision_request(url_query)

#     if not data or "objects" not in data:
#         return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     objects = data["objects"]

#     if len(objects) == 0:
#         return "No matching object in this application."

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     # Apply pagination
#     paginated_objects, pagination_info = paginate_results(objects, page)
    
#     objects_str = [generic_format_function(o) for o in paginated_objects]
    
#     # Build result with pagination info
#     result_parts = [
#         f"Objects in {application}:",
#         format_pagination_info(pagination_info)
#     ]
    
#     # Add search criteria summary
#     criteria = []
#     if name: criteria.append(f"name={name}")
#     if qualified_name: criteria.append(f"qualified_name={qualified_name}")
#     if file_path: criteria.append(f"file_path={file_path}")
#     if type: criteria.append(f"type={type}")
#     if criteria:
#         result_parts.append(f"Search criteria: {', '.join(criteria)}")
    
#     # Add navigation
#     if pagination_info["total_pages"] > 1:
#         nav = []
#         if pagination_info["has_previous"]:
#             nav.append(f"Previous: page={page-1}")
#         if pagination_info["has_next"]:
#             nav.append(f"Next: page={page+1}")
#         if nav:
#             result_parts.append(" | ".join(nav))
    
#     result_parts.append("")  # Empty line before results
#     result_parts.append("\n---\n".join(objects_str))
    
#     return "\n".join(result_parts)

@mcp.tool()
async def objects(application: str,
                  ctx: Context,
                  filters: str = None,
    page: int = 1) -> str:
    """Get objects in an application matching identification criteria. Paginated results.
       Filter objects using query string format: "field:operator:value,field:operator:value"

       Available fields for "contains" operator:
       name, fullname, mangling, type, filepath

       Examples:
       - Simple filter: "name:contains:myapp"
       - Multiple filters: "name:contains:web,type:contains:method"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1
    await ctx.report_progress(progress)


    url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
    criteria = {}
    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k in ['name','fullname','mangling','type','filepath'] and o == 'contains':
                    url_query += f"&{k}={quote(v)}" # Use quote for filter values
                    criteria[k] = v
                # Add other operators if needed
            # Handle invalid filter format?

    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)
    data = await vision_server._make_vision_request(url_query)

    if not data or "objects" not in data:
        return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    objects = data["objects"]

    if len(objects) == 0:
        return "No matching object in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Apply pagination
    paginated_objects, pagination_info = paginate_results(objects, page)

    objects_str = [generic_format_function(o) for o in paginated_objects]

    # Build extra info list for criteria
    criteria_info = []
    if criteria:
        criteria_info.append(f"Search criteria: {', '.join([f'{k}={v}' for k, v in criteria.items()])}")

    return format_paginated_result(
        base_message=f"Objects in {application}:",
        paginated_items_str=objects_str,
        pagination_info=pagination_info,
        extra_info=criteria_info
    )

# @mcp.tool()
# async def object_details(application: str, 
#                          ctx: Context,  
#                          id: str = None, 
#                          name: str = None, 
#                          qualified_name: str = None, 
#                          file_path: str = None, 
#                          type: str = None,
#     page: int = 1) -> str:
#     """Get details of object(s) in an application matching identification criteria.
#     - basic properties
#     - children objects
#     - inward and outward program flows (a.k.a., callers and callees)
#     - number of use in transactions (a.k.a. UI and API endpoints)
#     - number of use in data entity interaction networks (a.k.a. data graphs)
#     - code snippet (if available in Vision databases)
#     Paginated results.

#     Args:
#         application: string
#         name: string
#         qualified_name: string
#         file_path: string
#         type: string
#         id: string
#         page: int (default: 1)
#     """
#     progress = 1
    
#     objects = []
#     details = []

#     if id is not None and id != "": 
#         objects = [ { "id": id} ]
#         details = [ { "comment": f"Using object ID [{id}] to start the collection of their details"} ]
#     else:
#         url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
#         args_str = []
        
#         if name is not None and name != "":
#             url_query += f"&name={name}"
#             args_str.append(f"name={name}")

#         if qualified_name is not None and qualified_name != "":
#             url_query += f"&fullname={qualified_name}"
#             args_str.append(f"fullname={qualified_name}")

#         if file_path is not None and file_path != "":
#             url_query += f"&filepath={quote(file_path)}"
#             args_str.append(f"filepath={quote(file_path)}")

#         if type is not None and type != "":
#             url_query += f"&type={type}"
#             args_str.append(f"type={type}")

#         await ctx.report_progress(progress)
#         data = await vision_server._make_vision_request(url_query)

#         if not data or "objects" not in data:
#             return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]"

#         objects = data["objects"]
#         details = [ { "comment": f"Using query filters [{', '.join(args_str)}] to find {len(objects)} objects to start the collection of their details"} ]

#         if len(objects) == 0:
#             return f"No matching object in application {application}."

#     # Process objects with pagination
#     start_idx = (page - 1) * int(VISION_DISPLAY_PAGE_SIZE)
#     end_idx = min(start_idx + int(VISION_DISPLAY_PAGE_SIZE), len(objects))
#     objects_to_process = objects[start_idx:end_idx]
    
#     for i, o in enumerate(objects_to_process):
#         id = str(o['id'])
        
#         if vision_server._get_use_code():
#             url_query = f"applications/{application}/objects/{id}?select=source-locations"
#         else:
#             url_query = f"applications/{application}/objects/{id}"

#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         properties = await vision_server._make_vision_request(url_query)
        
        
#         if vision_server._get_use_code():
#             code_snippets = []
#             if properties and "sourceLocations" in properties:
#                 file_locations = properties["sourceLocations"]
#                 if file_locations:
#                     for location in file_locations:
#                         file_id = location['fileId']
#                         start_line = location['startLine']
#                         end_line = location['endLine']

#                         try:
#                             url_query = f"applications/{application}/files/{file_id}"
#                             url_query += f"?start-line={start_line}&end-line={end_line}" if int(start_line) > 0 and int(end_line) > 0 else ""
#                             code_snippet = await vision_server._make_vision_request(url_query, accept="text/plain;charset=utf-8")
#                         except:
#                             location["code_snippet"] = f"Error retrieving code snippet with [{vision_server._get_base_url()}/{url_query}]."
#                             code_snippets.append(location)

#                         if code_snippet:
#                             location["code_snippet"] = code_snippet
#                             code_snippets.append(location)
#                         else:
#                             location["code_snippet"] = f"No code snippet retrieved with [{vision_server._get_base_url()}/{url_query}]."
#                             code_snippets.append(location)  

#             all_code_snippets = {
#                 "nb_code_snippets": len(code_snippets),
#                 "code_snippets": code_snippets
#             }

#         url_query = f"applications/{application}/objects/{id}/properties"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         misc_properties = {
#             "misc_properties": await vision_server._make_vision_request(url_query)
#         }

#         url_query = f"applications/{application}/objects/{id}/documents"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         documents = {
#             "documents": await vision_server._make_vision_request(url_query)
#         }

#         url_query = f"applications/{application}/objects/{id}/callers"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         callers = {
#             "incoming_calls": await vision_server._make_vision_request(url_query)
#         }
        
#         url_query = f"applications/{application}/objects/{id}/callees"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         callees = {
#             "outgoing_calls": await vision_server._make_vision_request(url_query)
#         }
        
#         url_query = f"applications/{application}/objects/{id}/children"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         children = { 
#             "defines": await vision_server._make_vision_request(url_query) 
#         }
        
#         url_query = f"applications/{application}/objects/{id}/transactions"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         transactions = await vision_server._make_vision_request(url_query)
#         nb_transactions = { 
#             "nb_transactions":  len(transactions) if transactions is not None else 0,
#             "nb_transactions_comment": "use **transactions_using_object** Tool to get the actual list"
#         }

#         url_query = f"applications/{application}/objects/{id}/datagraphs"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         datagraphs = await vision_server._make_vision_request(url_query)
#         nb_datagraphs = { 
#             "nb_data_graphs":  len(datagraphs) if datagraphs is not None else 0,
#             "nb_data_graphs_comment": "use **data_graphs_involving_object** Tool to get the actual list"
#         }

#         details.append({
#             **properties,
#             **callers,
#             **callees,
#             **children,
#             **misc_properties,
#             **documents,
#             **nb_transactions,
#             **nb_datagraphs
#         })
#         if vision_server._get_use_code():
#             details.append({
#                 **all_code_snippets
#             })

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     # Create pagination info
#     total_pages = (len(objects) + int(VISION_DISPLAY_PAGE_SIZE) - 1) // int(VISION_DISPLAY_PAGE_SIZE)
#     pagination_info = {
#         "current_page": page,
#         "total_pages": total_pages,
#         "total_items": len(objects),
#         "showing_from": start_idx + 1 if objects_to_process else 0,
#         "showing_to": end_idx,
#         "has_next": page < total_pages,
#         "has_previous": page > 1
#     }
    
#     details_str = [generic_format_function(d) for d in details]
    
#     result = f"Object details for {application}:\n"
#     result += f"{format_pagination_info(pagination_info)}\n"
    
#     if pagination_info["total_pages"] > 1:
#         nav_hints = []
#         if pagination_info["has_previous"]:
#             nav_hints.append(f"Previous: page={page-1}")
#         if pagination_info["has_next"]:
#             nav_hints.append(f"Next: page={page+1}")
#         result += " | ".join(nav_hints) + "\n"
    
#     result += "\n" + "\n---\n".join(details_str)
#     return result

@mcp.tool()
async def object_details(application: str,
                         ctx: Context,
                         filters: str = None,
    page: int = 1) -> str:
    """Get details of object(s) in an application matching identification criteria.
    - basic properties
    - children objects
    - inward and outward program flows (a.k.a., callers and callees)
    - number of use in transactions (a.k.a. UI and API endpoints)
    - number of use in data entity interaction networks (a.k.a. data graphs)
    - code snippet (if available in Vision databases)
    Paginated results.
    Filter using query string format: "field:operator:value,field:operator:value"

    Available fields for "contains" operator:
    name, fullname, mangling, type, filepath

    Available fields for "eq" operator:
    id (id field has precedence over other filtering criteria)

    Examples:
    - Simple filter: "name:contains:myapp"
    - Multiple filters: "name:contains:web,type:contains:method"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1
    await ctx.report_progress(progress)

    objects = []
    details = []
    use_id = False
    id_filter = None
    criteria = {}

    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k == "id" and o == "eq":
                    use_id = True
                    id_filter = v
                    criteria = { "id": v}
                    break # ID filter takes precedence
                elif k in ['name','fullname','mangling','type','filepath'] and o == 'contains':
                    criteria[k] = v
            # Handle invalid filter format?

    if use_id:
        objects = [ { "id": id_filter} ]
        details = [ { "comment": f"Using object ID [{id_filter}] to start the collection of their details"} ]
    else:
        url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
        for k, v in criteria.items():
             url_query += f"&{k}={quote(v)}" # Use quote for filter values

        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)

        if not data or "objects" not in data:
            return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]"

        objects = data["objects"]
        details = [ { "comment": f"Using query filters [{', '.join([f'{k}={v}' for k, v in criteria.items()])}] to find {len(objects)} objects to start the collection of their details"} ]

        if len(objects) == 0:
            return f"No matching object in application {application}."

    # Process objects with pagination
    start_idx = (page - 1) * int(VISION_DISPLAY_PAGE_SIZE)
    end_idx = min(start_idx + int(VISION_DISPLAY_PAGE_SIZE), len(objects))
    objects_to_process = objects[start_idx:end_idx]

    # Collect details for the objects on the current page
    paginated_details = []
    for i, o in enumerate(objects_to_process):
        obj_id = str(o['id'])

        obj_details = {}

        # Fetch basic properties and source locations
        if vision_server._get_use_code():
            url_query = f"applications/{application}/objects/{obj_id}?select=source-locations"
        else:
            url_query = f"applications/{application}/objects/{obj_id}"

        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        properties = await vision_server._make_vision_request(url_query)
        if properties:
            obj_details.update(properties)

        # Fetch code snippets if available and enabled
        if vision_server._get_use_code() and properties and "sourceLocations" in properties:
            code_snippets = []
            file_locations = properties["sourceLocations"]
            if file_locations:
                for location in file_locations:
                    file_id = location.get('fileId')
                    start_line = location.get('startLine', 0)
                    end_line = location.get('endLine', 0)

                    if file_id:
                        try:
                            url_query = f"applications/{application}/files/{file_id}"
                            if int(start_line) > 0 and int(end_line) > 0:
                                url_query += f"?start-line={start_line}&end-line={end_line}"
                            code_snippet = await vision_server._make_vision_request(url_query, accept="text/plain;charset=utf-8")
                        except Exception as e:
                            code_snippet = f"Error retrieving code snippet with [{vision_server._get_base_url()}/{url_query}]: {e}"


                        loc_info = location.copy() # Copy location info
                        loc_info["code_snippet"] = code_snippet if code_snippet else f"No code snippet retrieved with [{vision_server._get_base_url()}/{url_query}]."
                        code_snippets.append(loc_info)

            obj_details["nb_code_snippets"] = len(code_snippets)
            obj_details["code_snippets"] = code_snippets


        # Fetch misc properties
        url_query = f"applications/{application}/objects/{obj_id}/properties"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        misc_properties = await vision_server._make_vision_request(url_query)
        if misc_properties:
             obj_details["misc_properties"] = misc_properties


        # Fetch documents
        url_query = f"applications/{application}/objects/{obj_id}/documents"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        documents = await vision_server._make_vision_request(url_query)
        if documents:
            obj_details["documents"] = documents


        # Fetch callers
        url_query = f"applications/{application}/objects/{obj_id}/callers"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        callers = await vision_server._make_vision_request(url_query)
        if callers:
            obj_details["incoming_calls"] = callers


        # Fetch callees
        url_query = f"applications/{application}/objects/{obj_id}/callees"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        callees = await vision_server._make_vision_request(url_query)
        if callees:
            obj_details["outgoing_calls"] = callees


        # Fetch children
        url_query = f"applications/{application}/objects/{obj_id}/children"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        children = await vision_server._make_vision_request(url_query)
        if children:
            obj_details["defines"] = children


        # Fetch transactions count
        url_query = f"applications/{application}/objects/{obj_id}/transactions"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        transactions = await vision_server._make_vision_request(url_query)
        obj_details["nb_transactions"] = len(transactions) if transactions is not None else 0
        obj_details["nb_transactions_comment"] = "use **transactions_using_object** Tool to get the actual list"


        # Fetch datagraphs count
        url_query = f"applications/{application}/objects/{obj_id}/datagraphs"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        datagraphs = await vision_server._make_vision_request(url_query)
        obj_details["nb_data_graphs"] = len(datagraphs) if datagraphs is not None else 0
        obj_details["nb_data_graphs_comment"] = "use **data_graphs_involving_object** Tool to get the actual list"

        paginated_details.append(obj_details)


    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Create pagination info (based on total objects found, not just those processed)
    total_pages = (len(objects) + int(VISION_DISPLAY_PAGE_SIZE) - 1) // int(VISION_DISPLAY_PAGE_SIZE)
    pagination_info = {
        "current_page": page,
        "total_pages": total_pages,
        "total_items": len(objects),
        "showing_from": start_idx + 1 if objects_to_process else 0,
        "showing_to": end_idx,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

    details_str = [generic_format_function(d) for d in paginated_details]

    # Build extra info list for criteria
    criteria_info = []
    if criteria:
        criteria_info.append(f"Search criteria: {', '.join(criteria)}")

    return format_paginated_result(
        base_message=f"Object details for {application}:",
        paginated_items_str=details_str,
        pagination_info=pagination_info,
        extra_info=criteria_info
    )

@mcp.tool()
async def add_object_document(application: str, ctx: Context,  id: str, text: str) -> str:
    """Add a document to a code element (a.k.a. object, artifact) identified by ID in an application.

    Args:
        application: string
        id: string
        text: string
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    if type(id) is int:
        id = str(id)

    url_query = f"applications/{application}/objects/{id}/document"
    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    
    print(f"Adding document to object {id} in application {application} with POST [{vision_server._get_base_url()}/{url_query}]")
    print(f"Document content: {text}")

    summary = await vision_server._make_vision_request(url_query, operation="post", body=text)

    if not summary:
        return f"Unable to fetch object with {id} in application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(summary) == 0:
        return f"No document added to object with ID {id} in application {application}."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    summary_str = [generic_format_function(s) for s in summary]
    return "\n---\n".join(summary_str)

@mcp.tool()
async def source_files(application: str, ctx: Context,  file_path: str,
    page: int = 1) -> str:
    """Find files which define code objects in an application matching identification criteria. Paginated results.
    (Simple "CONTAINS" match on provided file path, no wildcards or regex allowed)

    Args:
        application: string
        file_path: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1


    url_query = f"applications/{application}/source-files?filepath={quote(file_path)}"

    ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
    await ctx.report_progress(progress)
    files = await vision_server._make_vision_request(url_query)

    if not files:
        return f"Unable to fetch files defining code objects of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(files) == 0:
        return "No matching files defining code object in this application."

    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

     # Apply pagination
    paginated_files, pagination_info = paginate_results(files, page)
    
    files_str = "\n".join([generic_format_function(f) for f in paginated_files]).replace(
        VISION_ROOT, ""
    )
    
    # result = f"List of files defining code objects found:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{files_str}"
    # return result

    return format_paginated_result(
        base_message=f"List of files defining code objects found in {application} matching {file_path}:",
        paginated_items_str=files_str,
        pagination_info=pagination_info
    )


@mcp.tool()
async def source_file_details(application: str, 
                              ctx: Context,  
                              file_path: str, 
                              nature: Literal["inventory","intra","inward","outward","testing"] = "inventory",
    page: int = 1) -> str:
    """Get details about source file matching identification criteria in an application. Paginated results.
    Details on code elements defined by source file for selected nature:
    - inventory: all defined code elements
    - intra: dependencies between code elements defined in the same file
    - inward: incoming dependencies towards code elements defined in the file
    - outward: outgoing dependencies originating from code elements defined in the file
    - testing: elements to test if code elements defined in the file are evolved

    Args:
        application: string
        file_path: string
        nature: Literal["inventory","intra","inward","outward","testing"]
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    intro_text = f"Details ({nature}) about code elements defined in source files matching {file_path} in application:"
 
    url_query = f"applications/{application}/source-files?filepath={quote(file_path.replace(vision_server._get_root_dir(),""))}"

    await ctx.report_progress(progress)
    data = await vision_server._make_vision_request(url_query)

    if not data:
        return f"Unable to fetch files defining code objects of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(data) == 0:
        return "No matching files defining code object in this application."

    files = [d['file_path'].replace(vision_server._get_root_dir(),"") for d in data]

    file_details = []
    for i, f in enumerate(files):
        url_query = f"applications/{application}/source-file?relpath={quote(f)}&nature={nature}"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        details = await vision_server._make_vision_request(url_query)

        if details:
            file_details.append(details)

    # Apply pagination
    paginated_details, pagination_info = paginate_results(file_details, page)
    
    files_str = [generic_format_function(i) for i in paginated_details]

    # result = f"{intro_text}\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += f"\n{paginated_details}"
    # return result

    return format_paginated_result(
        base_message=f"List of files defining code objects found in {application} matching {file_path}:",
        paginated_items_str=files_str,
        pagination_info=pagination_info
    )

# @mcp.tool()
# async def file_dependencies(application: str, ctx: Context,  file_path: str) -> str:
#     """Get dependencies of file matching identification criteria in an application

#     Args:
#         application: string
#         file_path: string
#     """
#     progress = 1

#     intro_text = f"Details about source files with code elements calling code elements of source files matching {file_path} in application:"

#     url_query = f"applications/{application}/objects?limit={vision_server.page_size}&filepath={quote(file_path)}"

#     await ctx.report_progress(progress)
#     data = await vision_server._make_vision_request(url_query)

#     if not data or "objects" not in data:
#         return f"Unable to fetch files defining code objects of application {application} with [{vision_server._get_base_url()}/{url_query}]."

#     objects = data["objects"]

#     if len(objects) == 0:
#         return "No matching files defining code object in this application."

#     files = []
#     for o in objects:
#         file_path = o['filePath']
#         files.append(file_path)
        
#     files = list(set(files))

#     file_dependency_details = []
#     for i, f in enumerate(files):
#         url_query = f"applications/{application}/objects?limit={vision_server.page_size}&filepath={quote(f)}"
#         ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#         progress += 1
#         await ctx.report_progress(progress)
#         data = await vision_server._make_vision_request(url_query)

#         if data and "objects" in data:
#             for j, obj in enumerate(objects):
#                 url_query = f"applications/{application}/objects/{obj['id']}/callers"
#                 ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
#                 progress += 1
#                 await ctx.report_progress(progress)
#                 call_data = await vision_server._make_vision_request(url_query)

#                 if call_data:
#                     for c in call_data:
#                         file_dependency_details.append({ "filePath": obj['filePath'],**c})

#     ctx.info(f"Data reorganization for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     reorg_file_dependency_details = dict()
#     for d in file_dependency_details:
#         grouping_file = d['filePath']
#         if "sourceLocations" in d:
#             grouping_caller_locations = d['sourceLocations']
#             for location in grouping_caller_locations:
#                 grouping_caller_file = location['filePath'] if 'filePath' in location.keys() else location['fileName']

#                 if grouping_file in reorg_file_dependency_details.keys():
#                     grouped_file_dependecy_details = reorg_file_dependency_details[grouping_file]
#                 else:
#                     grouped_file_dependecy_details = dict()
            
#                 if grouping_caller_file in grouped_file_dependecy_details.keys():
#                     grouped_caller_file_details = grouped_file_dependecy_details[grouping_caller_file]
#                 else:
#                     grouped_caller_file_details = []

#                 call_info = {
#                     "to": obj['name'],
#                     "call_type": d['linkType'],
#                     "from": d['name']
#                 }
#                 grouped_caller_file_details.append(call_info)
#                 grouped_file_dependecy_details[grouping_caller_file] = grouped_caller_file_details
#                 reorg_file_dependency_details[grouping_file] = grouped_file_dependecy_details


#     # file_details_str =  [ generic_format_function(d) for d in file_details ]
#     # return  "\n---\n".join(file_details_str)

#     ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
#     progress += 1
#     await ctx.report_progress(progress)

#     # return f"Details about source files with code elements calling code elements of source files matching {file_path} in application:\n{[generic_format_function(d) for d in file_dependency_details]}"
#     return f"{intro_text}\n{generic_format_function(reorg_file_dependency_details)}"

@mcp.tool()
async def transactions_using_object(application: str, 
                                    ctx: Context,  
                                    filters: str = None,
    page: int = 1) -> str:
    """Get transations using object(s) in an application matching identification criteria. Paginated results.
    Filter objects using query string format: "field:operator:value,field:operator:value"

    Available fields for "contains" operator:
    name, fullname, mangling, type, filepath

    Available fields for "eq" operator:
    id (id field has precedence over other filtering criteria)

    Examples:
    - Simple filter: "name:contains:myapp"
    - Multiple filters: "name:contains:web,type:contains:method"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """

    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    objects = []
    transaction_details = []
    use_id = False
    id_filter = None
    criteria = {}

    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k == "id" and o == "eq":
                    use_id = True
                    id_filter = v
                    criteria = { "id": v}
                    break # ID filter takes precedence
                elif k in ['name','fullname','mangling','type','filepath'] and o == 'contains':
                    criteria[k] = v
            # Handle invalid filter format?

    if use_id:
        objects = [ { "id": id_filter} ]
        transaction_details = [ { "comment": f"Using object ID [{id}] to start the collection of the transactions using them"} ]
    else:
        url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
        for k, v in criteria.items():
             url_query += f"&{k}={quote(v)}" # Use quote for filter values


        progress += 1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)

        if not data or "objects" not in data:
            return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]"

        objects = data["objects"]
        transaction_details = [ { "comment": f"Using query filters [{', '.join([f'{k}={v}' for k, v in criteria.items()])}] to find {len(objects)} objects to start the collection of the transactions using them"} ]

        if len(objects) == 0:
            return f"No matching object in application {application}."

    # Process with pagination
    start_idx = (page - 1) * int(VISION_DISPLAY_PAGE_SIZE)
    end_idx = min(start_idx + int(VISION_DISPLAY_PAGE_SIZE), len(objects))
    objects_to_process = objects[start_idx:end_idx]

    for i, o in enumerate(objects_to_process):
        id = str(o['id'])
        
        url_query = f"applications/{application}/objects/{id}"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        properties = await vision_server._make_vision_request(url_query)
        
        url_query = f"applications/{application}/objects/{id}/transactions"
        progress += 1
        await ctx.report_progress(progress)
        transactions = await vision_server._make_vision_request(url_query)
        if transactions:
            transactions_json = { 
                "nb_transactions": len(transactions),
                "used_in_transactions":  transactions
            }
        else:
            transactions_json = { 
                    "nb_transactions": 0,
                    "used_in_transactions":  []
                }
        transaction_details.append({
            **properties,
            **transactions_json
        })


    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Create pagination info
    total_pages = (len(objects) + int(VISION_DISPLAY_PAGE_SIZE) - 1) // int(VISION_DISPLAY_PAGE_SIZE)
    pagination_info = {
        "current_page": page,
        "total_pages": total_pages,
        "total_items": len(objects),
        "showing_from": start_idx + 1 if objects_to_process else 0,
        "showing_to": end_idx,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

    details_str = [generic_format_function(d) for d in transaction_details]
    
    # result = f"Transactions using objects in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(details_str)
    # return result

    return format_paginated_result(
        base_message=f"Transactions using objects in {application}:",
        paginated_items_str=details_str,
        pagination_info=pagination_info
    )

@mcp.tool()
async def data_graphs_involving_object(application: str, 
                                       ctx: Context,  
                                       filters: str = None,
    page: int = 1) -> str:
    """Get data entity interaction networks, a.k.a. data graphs, involving object(s) in an application matching identification criteria. Paginated results.
    Filter objects using query string format: "field:operator:value,field:operator:value"

    Available fields for "contains" operator:
    name, fullname, mangling, type, filepath

    Available fields for "eq" operator:
    id (id field has precedence over other filtering criteria)

    Examples:
    - Simple filter: "name:contains:myapp"
    - Multiple filters: "name:contains:web,type:contains:method"

    Args:
        application: string
        filters: string
        page: int (default: 1)
    """
    logger.info(f"tool: {inspect.currentframe().f_code.co_name}({inspect.getargvalues(inspect.currentframe()).locals})")
    progress = 1

    objects = []
    datagraph_details = []
    use_id = False
    id_filter = None
    criteria = {}

    if filters:
        filter_list = filters.split(",")
        for f in filter_list:
            parts = f.split(":")
            if len(parts) == 3:
                k, o, v = parts
                if k == "id" and o == "eq":
                    use_id = True
                    id_filter = v
                    criteria = { "id": v}
                    break # ID filter takes precedence
                elif k in ['name','fullname','mangling','type','filepath'] and o == 'contains':
                    criteria[k] = v
            # Handle invalid filter format?

    if use_id:
        objects = [ { "id": id_filter} ]
        datagraph_details = [ { "comment": f"Using object ID [{id}] to start the collection of the datagraphs involving them"} ]
    else:
        url_query = f"applications/{application}/objects?limit={vision_server.page_size}"
        for k, v in criteria.items():
             url_query += f"&{k}={quote(v)}" # Use quote for filter values

        progress += 1
        await ctx.report_progress(progress)
        data = await vision_server._make_vision_request(url_query)

        if not data or "objects" not in data:
            return f"Unable to fetch objects of application {application} with [{vision_server._get_base_url()}/{url_query}]"

        objects = data["objects"]
        datagraph_details = [ { "comment": f"Using query filters [{', '.join([f'{k}={v}' for k, v in criteria.items()])}] to find {len(objects)} objects to start the collection of the data graphs involving them"} ]

        if len(objects) == 0:
            return f"No matching object in application {application}."

    # Process with pagination
    start_idx = (page - 1) * int(VISION_DISPLAY_PAGE_SIZE)
    end_idx = min(start_idx + int(VISION_DISPLAY_PAGE_SIZE), len(objects))
    objects_to_process = objects[start_idx:end_idx]

    for i, o in enumerate(objects_to_process):
        id = str(o['id'])
        
        url_query = f"applications/{application}/objects/{id}"
        ctx.info(f"Processing {url_query} for {ctx.request_id} of {ctx.client_id}")
        progress += 1
        await ctx.report_progress(progress)
        properties = await vision_server._make_vision_request(url_query)
        
        url_query = f"applications/{application}/objects/{id}/datagraphs"
        progress += 1
        await ctx.report_progress(progress)
        datagraphs = await vision_server._make_vision_request(url_query)
        if datagraphs:
            datagraphs_json = { 
                "nb_datagraphs": len(datagraphs),
                "involved_in_datagraphs":  datagraphs
            }
        else:
            datagraphs_json = { 
                    "nb_datagraphs": 0,
                    "involved_in_datagraphs":  []
                }
        datagraph_details.append({
            **properties,
            **datagraphs_json
        })


    ctx.info(f"Done processing for {ctx.request_id} of {ctx.client_id}")
    progress += 1
    await ctx.report_progress(progress)

    # Create pagination info
    total_pages = (len(objects) + int(VISION_DISPLAY_PAGE_SIZE) - 1) // int(VISION_DISPLAY_PAGE_SIZE)
    pagination_info = {
        "current_page": page,
        "total_pages": total_pages,
        "total_items": len(objects),
        "showing_from": start_idx + 1 if objects_to_process else 0,
        "showing_to": end_idx,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

    details_str = [generic_format_function(d) for d in datagraph_details]
    
    # result = f"Data graphs involving objects in {application}:\n"
    # result += f"{format_pagination_info(pagination_info)}\n"
    
    # if pagination_info["total_pages"] > 1:
    #     nav_hints = []
    #     if pagination_info["has_previous"]:
    #         nav_hints.append(f"Previous: page={page-1}")
    #     if pagination_info["has_next"]:
    #         nav_hints.append(f"Next: page={page+1}")
    #     result += " | ".join(nav_hints) + "\n"
    
    # result += "\n" + "\n---\n".join(details_str)
    # return result

    return format_paginated_result(
        base_message=f"Data graphs involving using objects in {application}:",
        paginated_items_str=details_str,
        pagination_info=pagination_info
    )


# PROMPTS ####

def PROMPT_TEMPLATE_TOOLS() -> str:
    return """
You can use the following tools to help you in code understanding and modification:
**objects** to find objects in a softawre application based on name, type, path
**object_details** to collect children, callers, callees, properties, code snippets of a code object
**transactions_using_object** to collect the transactions using the code object
**data_graphs_involving_object** to collect the data entity interaction networks (a.k.a. data graphs) involving the code object
**files** to find source files defining code objects in a softawre application based on file path
**file_details** to list code elements defined in a file in a software application based on file path
**transactions** to list transactions in a software application
**transaction_graph** to get nodes and links of a transaction
**data_graphs** to list data entity interaction networks (a.k.a. data graphs) in a software application
**data_graph_graph** to get nodes and links of a data entity interaction network (a.k.a. data graph)
**quality_insights** to list insights manifesting in an application
**quality_insight_occurrences** to list occurrences of objects in an application manifesting a specific insight
**architectural_graph** to list nodes or edges of the graph reprensenting the application architecture
**packages** to list external packages or libraries used to develop a software application
**package_interactions** to get details about how a software application uses external packages
"""

def PROMPT_TEMPLATE_UC_MOD() -> str:
    return """
You should use the tools:
**before** the modification you suggest: use context to better design the modification, trying - if not stated otherwise by the user - to minimize / avoid altogether breaking changes,
**after** the modification you suggest: use context to:
- at minimum, inform the user of 1. existing dependencies the user should take care of and 2. existing transactions that should be tested,
- ideally, suggest additional modifications to 1. repair the breaking changes, recursively in the call graph if needed, and 2. update test cases, run test cases (if possible)
"""

def PROMPT_TEMPLATE_UC_UND() -> str:
    return """
You should use the tools:
**when dealing with code elements** to understand direct then indirect dependencies and infer how the element(s) fit(s) in the software structure
**when dealing with whole applications** to undersand the overall architecture, the architecture of a sample of its transactions, ... 
"""

def PROMPT_TEMPLATE_UC_UND_FILE(application: str,  file_path: str) -> str:
    return f"""
You should use the tools:
**when dealing with code elements** to understand direct then indirect dependencies and infer how the element(s) fit(s) in the software structure
**when dealing with whole applications** to undersand the overall architecture, the architecture of a sample of its transactions, ... 

When dealing with {file_path} of application {application}:
1. start with files to check if it defines code elements, 
2. file_details to list the defined code elements.
"""

def PROMPT_TEMPLATE_UC_MOD_CVE(application: str,  cve: str) -> str:
    return f"""
As far as code modifications are concerned, you should always use the tools:
**before** the modification you suggest: use context to better design the modification, trying - if not stated otherwise by the user - to minimize / avoid altogether breaking changes,
**after** the modification you suggest: use context to:
- at minimum, inform the user of 1. existing dependencies the user should take care of and 2. existing transactions that should be tested,
- ideally, suggest additional modifications to 1. repair the breaking changes, recursively in the call graph if needed, and 2. update test cases, run test cases (if possible)

When dealing with {cve} of application {application}:
1. start with get_quality_insight_occurrences to find code objects manifesting the CVE
2. use object_details to know more about the code objects to fix/replace/...
3. use object_details on dependent code objects to check/fix/replace/...
4. use transactions_using_object on all the updated code objects and consolidate the different transactions that will have to be tested after the changes.
5. use datagraphs_involving_object on all the updated code objects and consolidate the different datagraphs that will have to be checked after the changes.
tip: use packages to get external packages and libraries already in use in the application to get ideas for easier fix
"""


@mcp.prompt()
def vision_context_for_code_modification() -> list[base.Message]:
    """Help update software code elements and structures with enriched context"""
    tool_text = PROMPT_TEMPLATE_TOOLS()
    uc_text = PROMPT_TEMPLATE_UC_MOD()
    return [
        base.UserMessage(tool_text),
        base.UserMessage(uc_text),
    ]

@mcp.prompt()
def vision_context_for_code_understanding() -> list[base.Message]:
    """Help understand software code elements and structures with enriched context"""
    tool_text = PROMPT_TEMPLATE_TOOLS()
    uc_text = PROMPT_TEMPLATE_UC_UND()
    return [
        base.UserMessage(tool_text),
        base.UserMessage(uc_text),
    ]

@mcp.prompt()
def vision_context_for_file_understanding(application: str, file_path: str) -> list[base.Message]:
    """Help understand software code elements defined in file with enriched context"""
    tool_text = PROMPT_TEMPLATE_TOOLS()
    uc_text = PROMPT_TEMPLATE_UC_UND_FILE(application=application,file_path=file_path)
    return [
        base.UserMessage(tool_text),
        base.UserMessage(uc_text),
    ]

@mcp.prompt()
def vision_context_for_cve_remediation(application: str, cve: str) -> list[base.Message]:
    """Help understand software code elements defined in file with enriched context"""
    tool_text = PROMPT_TEMPLATE_TOOLS()
    uc_text = PROMPT_TEMPLATE_UC_MOD_CVE(application=application,cve=cve)
    return [
        base.UserMessage(tool_text),
        base.UserMessage(uc_text),
    ]

# RESOURCES ####

@mcp.resource("vision://config")
def config() -> str:
    return f"Targeting {vision_server._get_base_url()} API endpoint"

@mcp.resource("vision://applications")
async def applications() -> str:
    """Available applications"""
    url_query = f"applications"
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_str = "\n---\n".join([generic_format_function(a) for a in applications])
    return f"Available applications:\n{application_str}"

@mcp.resource("vision://applications/transactions")
async def applications_transactions() -> str:
    """Applications' transactions"""
    url_query = f"applications"
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/transactions"
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            transactions = {
                "nb_transactions": len(data),
                "transactions": data
            }
        else:
            transactions = {
                "nb_transactions": 0
            }

        application_details.append({
            **info,
            **transactions
        })


    application_str = "\n---\n".join([generic_format_function(d) for d in application_details])
    return f"Applications' transactions:\n{application_str}"

@mcp.resource("vision://applications/quality_insights")
async def applications_quality_insights() -> str:
    """Applications' quality insights"""
    url_query = f"applications"
    applications = await vision_server._make_vision_request(url_query)

    if not applications:
        return f"Unable to fetch applications with [{vision_server._get_base_url()}/{url_query}]."

    if len(applications) == 0:
        return "No applications available."
    
    application_details = []
    for i, info in enumerate(applications):
        application = info["name"]

        url_query = f"applications/{application}/insights/cve"
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            cves = {
                "nb_cves": len(data),
                "cves": data
            }
        else:
            cves = {
                "nb_cves": 0
            }

        url_query = f"applications/{application}/insights/cloud-detection-patterns"
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            cloud = {
                "nb_cloud_detection_patterns": len(data),
                "cloud_detection_patterns": data
            }
        else:
            cloud = {
                "nb_cloud_detection_patterns": 0
            }

        url_query = f"applications/{application}/insights/green-detection-patterns"
        data = await vision_server._make_vision_request(url_query)
        if data and len(data) >0:
            green = {
                "nb_green_detection_patterns": len(data),
                "green_detection_patterns": data
            }
        else:
            green = {
                "nb_green_detection_patterns": 0
            }

        application_details.append({
            **info,
            **cves,
            **cloud,
            **green
        })


    application_str = "\n---\n".join([generic_format_function(d) for d in application_details])
    return f"Applications' quality insights:\n{application_str}"

@mcp.resource("vision://applications/{application}")
async def application_resource(application: str) -> str:
    """Application resources (transaction list for now).

    Args:
        application: string
    """
    url_query = f"applications/{application}/transactions?limit={vision_server.page_size}"
    transactions = vision_server._make_vision_request(url_query)

    if not transactions:
        return f"Unable to fetch transactions of application {application} with [{vision_server._get_base_url()}/{url_query}]."

    if len(transactions) == 0:
        return "No transactions in this application."

    transactions_str = "\n---\n".join([generic_format_function(t) for t in transactions])
    return f"Transactions of application {application}:\n{transactions_str}"

# @mcp.list_resource_templates

def main():
    
    mcp.run(
      transport="stdio"
    )


if __name__ == "__main__":
    main()


