from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from .utils import ensure_dir,safe_filename

def bar(df,x,y,title,path,top=20,horizontal=True):
    d=df.head(top).copy()
    plt.figure(figsize=(10,6))
    if horizontal:
        d=d.iloc[::-1]; plt.barh(d[x].astype(str),d[y]); plt.xlabel(y); plt.ylabel(x)
    else: plt.bar(d[x].astype(str),d[y]); plt.xticks(rotation=45,ha="right"); plt.ylabel(y); plt.xlabel(x)
    plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=300,bbox_inches="tight"); plt.close()

def line(df,x,y,title,path):
    plt.figure(figsize=(10,5)); plt.plot(df[x],df[y],marker="o"); plt.xlabel(x); plt.ylabel(y); plt.title(title); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def network_plot(G,path,title):
    plt.figure(figsize=(12,10))
    if G.number_of_nodes()==0: plt.text(.5,.5,"Insufficient data",ha="center")
    else:
        pos=nx.spring_layout(G,seed=42,k=1/max(1,G.number_of_nodes()**.5))
        sizes=[80+G.nodes[n].get("frequency",1)*35 for n in G]
        widths=[.3+G[u][v].get("weight",1)*.15 for u,v in G.edges()]
        nx.draw_networkx_nodes(G,pos,node_size=sizes,alpha=.8)
        nx.draw_networkx_edges(G,pos,width=widths,alpha=.25)
        labels={n:n for n in sorted(G,key=lambda n:G.nodes[n].get("frequency",1),reverse=True)[:25]}
        nx.draw_networkx_labels(G,pos,labels,font_size=7)
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(path,dpi=300,bbox_inches="tight"); plt.close()
