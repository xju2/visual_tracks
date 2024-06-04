import ROOT
from array import array

def graph(name, x, y):
    """
    x -- a list of x-values
    y -- a list of y-values
    """
    gr = ROOT.TGraph(len(x), array('f', x), array('f', y))
    gr.SetName(name)
    return gr


def graph_error(name, x, xe, y, ye):
    """ Symmetric errors
    x -- a list of x-values
    xe -- a list of x-errors
    y -- a list of y-values
    ye -- a list of y-errors
    """
    gr = ROOT.TGraphErrors(
        len(x), array('f', x), array('f', y),
        array('f', xe), array('f', ye)
    )
    gr.SetName(name)
    return gr


def unequal_bin_hist(hist_name, bin_list):
    """ create TH1F using a list as x-axis
    """
    nbins = len(bin_list) - 1
    h1 = ROOT.TH1F(hist_name, hist_name, nbins, array('f', bin_list))
    return h1
