#!/usr/bin/env python -*- coding: utf-8 -*-

'''This module allows to import and export
(http://graph-tool.skewed.de)[graph-tool] Graph objects to HDF5 files using pandas
'''

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import graph_tool.all as gt
import pandas as pd


TYPES_WHITELIST = ['bool', 'uint8_t', 'double', 'float',
                   'int16_t', 'int32_t', 'int64_t',
                   'short', 'int', 'long', 'long long',
                   'long double']
TYPES_BLACKLIST = ['string', 'vector<bool>', 'vector<uint8_t>',
                   'vector<int16_t>', 'vector<short>',
                   'vector<int32_t>', 'vector<int>',
                   'vector<int64_t>', 'vector<long>',
                   'vector<int64_t>', 'vector<long>',
                   'vector<long long>', 'vector<double>', 'vector<float>',
                   'vector<long double>', 'vector<string>',
                   'python::object', 'object']

ALIASES = {'float64':'double', 'int8':'bool', 'int32':'int'}



def graph_to_dataframes(graph, stamp=None):
    '''Function packs the vertex and edge propery maps of
    an **unfiltered** gt.Graph object in two dataframes.

    parameters:
    ===========

    graph: a `gt.Graph` object whose 'internal' property maps will be output.
    
    stamp: an integer or a time stamp, default `None`. If `stamp` is
        not None, it will be used as a supplementary index, this is
        usefull to store a dynmical graph history.

    returns:
    ========

    vertex_df : a pd.DataFrame containing the graph's vertex
        PropertyMaps as columns.  If `stamp` is `None`, the DataFrame
        index corresponds to the graph's vertices indices, else it
        is a pd.MultiIndex with (stamp, index) pairs.
    
    edge_df : a pd.DataFrame containing the graph's edge PropertyMaps
        as columns. If `stamp` is `None`, the DataFrame index is a
        pd.MultiIndex with `(source, target)` pairs, where `source`
        and `target` are the source and target vertices of a given edge.

    note:
    =====

    only the following value types are supported: 'bool', 'uint8_t',
    'double', 'float', 'int16_t', 'int32_t', 'int64_t', 'short',
    'int', 'long', 'long long', 'long double'
    The type 'string' might work but is untested. There might be a way to store
    vectors using `Pannels` object, but this is also untested.
    '''

    ## TODO: print out which propery maps were not ported
    if stamp is not None:
        vertex_index = pd.MultiIndex.from_tuples([(stamp, graph.vertex_index[v])
                                                  for v in graph.vertices()],
                                                 names=('stamp', 'vertex_index'))
        edge_index = pd.MultiIndex.from_tuples([(stamp, graph.vertex_index[source],
                                                 graph.vertex_index[target])
                                                for source, target in graph.edges()],
                                                names=('stamp', 'source', 'target'))
    else:
        vertex_index = pd.Index([graph.vertex_index[v]
                                 for v in graph.vertices()],
                                name='vertex_index')
        edge_index = pd.MultiIndex.from_tuples([(graph.vertex_index[source],
                                                 graph.vertex_index[target])
                                                for source, target in graph.edges()],
                                                names=('source', 'target'))
        
    vertex_df = pd.DataFrame({key: np.array(prop.a, dtype=prop.a.dtype)
                              for key, prop in graph.vertex_properties.items()
                              if prop.value_type() in TYPES_WHITELIST},
                             index=vertex_index)

    edge_df = pd.DataFrame({key: np.array(prop.a, dtype=prop.a.dtype)
                            for key, prop in graph.edge_properties.items() 
                            if prop.value_type() in TYPES_WHITELIST},
                           index=edge_index)
    return vertex_df, edge_df

def frames_to_hdf(vertex_df, edge_df, fname, reset=False, **kwargs):
    '''
    Records the two  DataFrame in the hdf file filename
    '''
    with pd.get_store(fname) as store:
        if not len(store.keys()):
            store.put('edges', edge_df,
                      format='table', **kwargs)
            store.put('vertices', vertex_df,
                      format='table', **kwargs) 
        elif reset:
            try:
                store.remove('vertices')
            except KeyError:
                pass
            try:
                store.remove('edges')
            except KeyError:
                pass
            store.put('vertices', vertex_df,
                      format='table', **kwargs) 
            store.put('edges', edge_df,
                      format='table', **kwargs)
        else:
            store.append('vertices', vertex_df,
                         format='table', **kwargs) 
            store.append('edges', edge_df,
                         format='table', **kwargs)
           
def graph_to_hdf(graph, fname, stamp=None, reset=False, **kwargs):
    '''
    
    '''
    vertex_df, edge_df = graph_to_dataframes(graph, stamp=stamp)
    frames_to_hdf(vertex_df, edge_df, fname, **kwargs)

def graph_from_dataframes(vertex_df, edge_df):
    '''Re-creates a Graph object with PropertyMaps taken from the vertex_df and edge_df DataFrames

    Paramters:
    ==========
    verex_df: a DataFrame with an index named 'vertex_index'    
    edge_df: a DataFrame with a multi-index named ('source', 'target')

    Returns:
    ========
    graph: a grah-tool Graph with PropertyMaps copied from the columns of the input DataFrames
    '''

    graph = gt.Graph(directed=True)
        
    vertex_index = vertex_df.index.get_level_values(level='vertex_index')
    vertices = graph.add_vertex(n=vertex_index.shape[0])
    for col in vertex_df.columns:
        dtype = ALIASES[vertex_df[col].dtype.name]
        prop = graph.new_vertex_property(dtype)
        prop.a = vertex_df[col]
        graph.vertex_properties[col] = prop

    src = edge_df.index.names.index('source')
    trgt = edge_df.index.names.index('target')
    for tup in edge_df.index:
        source, target = tup[src], tup[trgt]
        edge = graph.add_edge(source, target)

    for col in edge_df.columns:
        dtype = ALIASES[edge_df[col].dtype.name]
        prop = graph.new_edge_property(dtype)
        prop.a = edge_df[col]
        graph.edge_properties[col] = prop

    return graph


def frames_from_hdf(fname, stamp=None):
    
    with pd.get_store(fname) as store:
        if stamp is not None:
            vertex_df = store.select('vertices', where="'stamp'=stamp")
            edge_df = store.select('edges', where="'stamp'=stamp")
        else:
            vertex_df = store['vertices']
            edge_df = store['edges']
    return vertex_df, edge_df

def graph_from_hdf(fname, stamp=None):

    vertex_df, edge_df = frames_from_hdf(fname, stamp=None)
    return graph_from_dataframes(vertex_df, edge_df)