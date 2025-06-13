import pytest
from src.core.interconnect import InterconnectNetwork, InterconnectConfig, InterconnectTopology, Message, MessageType


def xy_next_hop(curr, dest):
    curr_x, curr_y = curr
    dest_x, dest_y = dest
    if curr_x != dest_x:
        return (curr_x + 1, curr_y) if curr_x < dest_x else (curr_x - 1, curr_y)
    if curr_y != dest_y:
        return (curr_x, curr_y + 1) if curr_y < dest_y else (curr_x, curr_y - 1)
    return curr


def create_network(rows=3, cols=3):
    cfg = InterconnectConfig(topology=InterconnectTopology.MESH)
    return InterconnectNetwork(cfg, (rows, cols))


def test_routing_table_xy():
    net = create_network(3, 3)
    for router_id, router in net.routers.items():
        for x in range(3):
            for y in range(3):
                dest = (x, y)
                expected = xy_next_hop(router_id, dest)
                assert router.routing_table[dest] == expected
                msg = Message(0, MessageType.DATA, router_id, dest, payload_bits=8)
                assert router.route_message(msg) == expected
