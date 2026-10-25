"""Graph views of an observation.

Converts the observation matrix of :class:`nasimemu_enhanced.env.NASimEmuEnv`
into a graph of hosts and subnets, lists the actions possible on the observed
hosts and plots the observed network.
"""
from typing import Any
from nasimemu_enhanced.simulation.host_vector import HostVector
import numpy as np
from numpy.typing import NDArray

import networkx as nx
import plotly.graph_objects as go

def get_possible_actions(env: Any, s: NDArray[Any]) -> list[tuple[int, int, int]]:
    """List every action on every host in the observation.

    Parameters
    ----------
    env : NASimEmuEnv
        the environment, used for its ``action_list``
    s : numpy.ndarray
        observation as a matrix

    Returns
    -------
    list of tuple
        ``(subnet, host, action_id)`` for every observed host and every
        action, without the terminal action. Whether an action can succeed is
        not considered.
    """
    # gather all possible addresses
    addresses = []
    for host_data in s[:-1]:
        host_vector = HostVector(host_data)
        addresses.append(host_vector.address)

    possible_actions = []

    for host_add in addresses:
        possible_actions.extend([(int(host_add[0]), int(host_add[1]), action_id) for action_id in range(len(env.action_list))])

    return possible_actions

def _complete_graph(n: Any) -> list[tuple[Any, Any]]:
    """Return the edges, in both directions, between all pairs of the nodes."""
    return [(a, b) for a in n for b in n if a != b]

def _gen_edge_index_v1(node_index: NDArray[Any], subnets: list[Any], subnet_graph: Any) -> NDArray[Any]:
    """Generate the edges of graph version 1.

    All nodes of a subnet, its hosts and the subnet node itself, are connected
    to each other. Two subnet nodes are connected if one subnet was discovered
    from the other.

    Parameters
    ----------
    node_index : numpy.ndarray
        ``(subnet, host)`` of every node, hosts first, then the subnet nodes
        with host = -1
    subnets : list of int
        the discovered subnets, in the order of their nodes
    subnet_graph : set of tuple
        ``(from, to)`` pairs of subnets

    Returns
    -------
    numpy.ndarray
        edges as an array of shape (2, number of edges) holding node
        positions; every connection is listed in both directions
    """
    edge_index = []

    # hosts in subnets are connected together
    for subnet in subnets:
        hosts_in_subnet = np.flatnonzero( node_index[:, 0] == subnet )
        edge_index.append( _complete_graph(hosts_in_subnet) )

    # all subnets together
    # sub_index = np.flatnonzero( node_index[:, 1] == -1 )
    # if len(sub_index) > 1:
    #     edge_index.append( _complete_graph(sub_index) )

    host_len = len(node_index) - len(subnets)

    for subnet_edge in subnet_graph:
        s_from = subnets.index(subnet_edge[0]) + host_len
        s_to   = subnets.index(subnet_edge[1]) + host_len

        edge_index.append([(s_from, s_to), (s_to, s_from)])

    result: NDArray[Any] = np.concatenate(edge_index).T

    return result

def _gen_edge_index_v2(node_index: NDArray[Any], subnets: list[Any], subnet_graph: Any) -> NDArray[Any]:
    """Generate the edges of graph version 2.

    Every host is connected to the node of its subnet only, not to the other
    hosts. Two subnet nodes are connected if one subnet was discovered from
    the other.

    Parameters
    ----------
    node_index : numpy.ndarray
        ``(subnet, host)`` of every node, hosts first, then the subnet nodes
        with host = -1
    subnets : list of int
        the discovered subnets, in the order of their nodes
    subnet_graph : set of tuple
        ``(from, to)`` pairs of subnets

    Returns
    -------
    numpy.ndarray
        edges as an array of shape (2, number of edges) holding node
        positions; every connection is listed in both directions
    """
    edge_index = []

    # hosts in subnets are connected to their subnet
    for subnet in subnets:
        hosts_in_subnet = np.flatnonzero( node_index[:, 0] == subnet )
        subnet_node = np.flatnonzero( (node_index[:, 0] == subnet) * (node_index[:, 1] == -1))[0]

        edge_index.append( [(x, subnet_node) for x in hosts_in_subnet if x != subnet_node] )
        edge_index.append( [(subnet_node, x) for x in hosts_in_subnet if x != subnet_node] )
        # edge_index.append( _complete_graph(hosts_in_subnet) )

    # all subnets together
    # sub_index = np.flatnonzero( node_index[:, 1] == -1 )
    # if len(sub_index) > 1:
    #     edge_index.append( _complete_graph(sub_index) )

    host_len = len(node_index) - len(subnets)

    for subnet_edge in subnet_graph:
        s_from = subnets.index(subnet_edge[0]) + host_len
        s_to   = subnets.index(subnet_edge[1]) + host_len

        edge_index.append([(s_from, s_to), (s_to, s_from)])

    result: NDArray[Any] = np.concatenate(edge_index).T
    return result

def convert_to_graph(s: NDArray[Any], subnet_graph: Any, version: int = 1) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Convert an observation matrix into a graph of hosts and subnets.

    Every observed host and every subnet with an observed host becomes a
    node. The row with the action result is dropped.

    Parameters
    ----------
    s : numpy.ndarray
        observation as a matrix
    subnet_graph : set of tuple
        ``(from, to)`` pairs of subnets, recording from which subnet another
        one was discovered
    version : int, optional
        1 connects all nodes of a subnet to each other, 2 connects the hosts
        only to the node of their subnet (default=1)

    Returns
    -------
    numpy.ndarray
        node features, one row per node, hosts first. A host node is its row
        of the observation, a subnet node has a 1 in the column of its subnet;
        both have the node type prepended (0 = host, 1 = subnet).
    numpy.ndarray
        edges, of shape (2, number of edges), holding node positions
    numpy.ndarray
        ``(subnet, host)`` of every node, with host = -1 for subnet nodes
    numpy.ndarray
        position of every host node for a positional encoding, 0 for subnet
        nodes

    Raises
    ------
    NotImplementedError
        if the version is neither 1 nor 2
    """
    # hosts_discovered = s[:-1, HostVector._discovered_idx] == 1
    # host_feats = s[:-1][hosts_discovered]

    host_feats = s[:-1] # skip the result row
    host_index = np.array([HostVector(x).address for x in host_feats])
    # host_index = self.host_index[hosts_discovered]

    subnets_discovered = list( {x[0] for x in host_index} )
    subnet_index = np.array( [(x, -1) for x in subnets_discovered])
    # subnet_index = self.subnet_index[subnets_discovered]
    subnet_feats = np.eye(max(subnets_discovered)+1, host_feats.shape[1])[subnets_discovered]

    # subnet 0 is internet, and will always be empty

    node_type = np.zeros( (len(host_feats) + len(subnet_feats), 1) )
    node_type[len(host_feats):] = 1     # 0 for hosts; 1 for subnets

    node_feats = np.concatenate( [host_feats, subnet_feats] )
    node_feats = np.concatenate( [node_type, node_feats], axis=1 )

    node_index = np.concatenate( [host_index, subnet_index] )
    pos_index = np.concatenate( [np.arange(len(host_index)), np.zeros(len(subnet_index))] ) # for positional encoding

    # create edge index
    if version == 1:
        edge_index = _gen_edge_index_v1(node_index, subnets_discovered, subnet_graph)
    elif version == 2:
        edge_index = _gen_edge_index_v2(node_index, subnets_discovered, subnet_graph)
    else:
        raise NotImplementedError()

    return node_feats, edge_index, node_index, pos_index

# inspired from https://plotly.com/python/network-graphs/
def _plot(G: Any) -> Any:
    """Draw a graph prepared by :func:`_make_graph`.

    Connections between subnets are drawn solid, all others dashed.

    Returns
    -------
    plotly.graph_objects.Figure
        the figure
    """
    def get_edges(type: str) -> tuple[list[Any], list[Any]]:
        edge_x = []
        edge_y = []

        for edge in G.edges():
            node_in  = G.nodes[edge[0]]
            node_out = G.nodes[edge[1]]

            if type == 'subnet': # both have to be subnets
                if not (node_in['type'] == 'subnet' and node_out['type'] == 'subnet'):
                    continue

            if type == 'node':  # both can't be subnets
                if node_in['type'] == 'subnet' and node_out['type'] == 'subnet':
                    continue

            x0, y0 = node_in['pos']
            x1, y1 = node_out['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        return edge_x, edge_y

    node_edges = get_edges(type='node')
    subnet_edges = get_edges(type='subnet')

    edge_trace = go.Scatter(
        x=node_edges[0], y=node_edges[1],
        line=dict(width=1, color='black', dash='dash'),
        hoverinfo='none',
        mode='lines')

    subnet_trace = go.Scatter(
        x=subnet_edges[0], y=subnet_edges[1],
        line=dict(width=1, color='black'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_symbols = []
    node_line_widths = []

    for node_id, node in G.nodes.items():
        x, y = node['pos']
        node_x.append(x)
        node_y.append(y)

        node_text.append(f"{node['label']}")
        node_color.append(node['color'])
        node_symbols.append(node['symbol'])
        node_line_widths.append(node['line_width'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(showscale=False, color=node_color, symbol=node_symbols, size=40, line_width=node_line_widths, line_color="black"),
        text=node_text,
        # text_color="black",
        textposition="top center")

    fig = go.Figure(data=[edge_trace, subnet_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            # family="Courier New, monospace",
            size=25,
            color="black"
        )
    )

    return fig

def _make_graph(s: Any, a: Any) -> Any:
    """Build a networkx graph with everything needed to draw it.

    Every node gets a position, a label, a colour and a symbol. Subnets are
    grey triangles, hosts are circles, red if they are sensitive. A host label
    shows its address, coloured by the access gained, and its running
    services. The target of the last action is outlined and labelled with the
    name of the action.

    Parameters
    ----------
    s : tuple
        the graph as returned by :func:`convert_to_graph`
    a : Action or TerminalAction
        the last action

    Returns
    -------
    networkx.Graph
        the graph with the display attributes set on its nodes
    """
    node_feats, edge_index, node_index, pos_index = s

    G: nx.Graph[Any] = nx.Graph()
    G.add_edges_from(edge_index.T)
    pos = nx.kamada_kawai_layout(G)

    a_target = (-1, -1) if a.__class__.__name__ == "TerminalAction" else a.target

    def get_host_conf(host_id: int) -> str:
        host_vec = HostVector(node_feats[host_id][1:])

        running_services = []
        for srv_name in HostVector.service_idx_map:
            if host_vec.is_running_service(srv_name):
                running_services.append(srv_name.split('_')[-1])

        service_str = ", ".join(running_services)
        if service_str:
            service_str = "<br>" + service_str

        return service_str

    def get_host_string(i: int) -> str:
        node_idx = node_index[i]
        host_conf = get_host_conf(i)

        node_action = a.name if np.all(np.array(a_target) == np.array(node_idx)) else ""

        access_colors = ['black', 'orange', 'red']
        node_str = f"<span style='color:{access_colors[is_host_controlled(i)]};'>{node_idx}</span>"

        return f"{node_str} <b>{node_action}</b>{host_conf}"

    def is_host_sensitive(i: int) -> bool:
        host_vec = HostVector(node_feats[i][1:])
        return bool(host_vec.value > 0)

    def is_host_controlled(i: int) -> int:
        host_vec = HostVector(node_feats[i][1:])
        return int(host_vec.access)

    def get_node_color(i: int) -> str:
        if node_index[i][1] == -1:
            return 'grey'
        else:
            return 'red' if is_host_sensitive(i) else 'white'


    node_labels = {i: f"Subnet {node_index[i][0]}" if node_index[i][1] == -1 else get_host_string(i) for i in G.nodes}
    node_types = {i: 'subnet' if node_index[i][1] == -1 else 'node' for i in G.nodes}
    node_colors = {i: get_node_color(i) for i in G.nodes}
    node_symbols = {i: 'triangle-up' if node_index[i][1] == -1 else 'circle' for i in G.nodes}
    node_line_widths = {i: 6.0 if np.all(np.array(a_target) == np.array(node_index[i])) else 1.0 for i in G.nodes}

    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_types, 'type')
    nx.set_node_attributes(G, node_colors, 'color')
    nx.set_node_attributes(G, node_symbols, 'symbol')
    nx.set_node_attributes(G, node_line_widths, 'line_width')

    return G

def plot_network(s: NDArray[Any], subnet_graph: Any, last_action: Any) -> Any:
    """Plot the observed network and the last action.

    Parameters
    ----------
    s : numpy.ndarray
        observation as a matrix, e.g. ``env.s_raw``
    subnet_graph : set of tuple
        ``(from, to)`` pairs of subnets, e.g. ``env.subnet_graph``
    last_action : Action or TerminalAction
        the last action, e.g. ``env.a_raw``; its target is highlighted

    Returns
    -------
    plotly.graph_objects.Figure
        the figure; show it with ``fig.show()`` or save it with
        ``fig.write_image()``
    """
    s_graph = convert_to_graph(s, subnet_graph, version=2)
    G = _make_graph(s_graph, last_action)
    fig = _plot(G)
    
    # use:
    # fig.show()

    # or:
    # pio.kaleido.scope.mathjax = None
    # fig.write_image(f"trace.pdf", width=1200, height=600, scale=1.0)

    return fig