
import argparse
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import colormaps

LABEL_SIZE=19
TICK_SIZE=16
plt.rcParams.update({"font.size":TICK_SIZE,"axes.labelsize":LABEL_SIZE,"xtick.labelsize":TICK_SIZE,"ytick.labelsize":TICK_SIZE})

def parse_comp_params(series):
    return [ast.literal_eval(s) for s in series.unique()]

def key_from_dict(d):
    return (d["lambda_lines_2d"],d["lambda_points_line_2d"],d["lambda_clutter"])

def build_global_color_map(all_keys):
    cmap=colormaps["tab20"]
    palette=cmap.colors
    return {k:palette[i%len(palette)] for i,k in enumerate(sorted(all_keys))}

def wls_fit(ddf):
    x=ddf["dataset_size"].astype(float).values
    y=ddf["frechet_distance"]["mean"].astype(float).values
    w=np.sqrt(x)
    X=sm.add_constant(1.0/x)
    model=sm.WLS(y,X,weights=w).fit()
    yhat=model.predict(X)
    return x,yhat

def collect_color_keys_across_experiments(root,exp_ids):
    keys=set()
    for exp_id in exp_ids:
        csv_path=Path(root)/exp_id/"results_cumalative_sampling.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")
        data=pd.read_csv(csv_path)
        all_comp_dicts=parse_comp_params(data["comparison_params"])
        keys|=set(key_from_dict(d) for d in all_comp_dicts)
    return keys

def intersect_feature_dims(root,exp_ids,requested):
    intersection=None
    for exp_id in exp_ids:
        csv_path=Path(root)/exp_id/"results_cumalative_sampling.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")
        data=pd.read_csv(csv_path)
        dims=set(map(int,set(data["feature_dim"].unique())))
        intersection = dims if intersection is None else (intersection & dims)
    if requested is None:
        return sorted(intersection)
    req=set(map(int,requested))
    return sorted(intersection & req)

def plot_frd_vs_n(ax,df_fd,color_map,legend):
    
    groups=list(df_fd.groupby("comparison_params"))
    
    for comp_params,df in groups:
        comp_gen=ast.literal_eval(comp_params)
        k=key_from_dict(comp_gen)
        col=color_map.get(k,(0.3,0.3,0.3,1.0))
        ddf=df.groupby("dataset_size",as_index=False).describe().sort_values("dataset_size")
        if ddf.shape[0]==0:
            continue
        x=ddf["dataset_size"].astype(float).values
        y=ddf["frechet_distance"]["mean"].astype(float).values
        ax.scatter(x,y,color=col,s=55,alpha=0.9,label=str(k) if legend else None)
        try:
            xhat,yhat=wls_fit(ddf)
            order=np.argsort(xhat)
            ax.plot(xhat[order],yhat[order],"--",color=col,lw=2.0,alpha=0.9)
        except Exception:
            pass
    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks=np.unique(df_fd["dataset_size"].values)
    try:
        ax.set_xticks(xticks,[f"{int(v):d}" for v in xticks], rotation=45)
    except Exception:
        pass

    ax.grid(True,alpha=0.3)
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"$\text{FRD}_\infty$")

def plot_experiment(root,exp_id,color_map,feature_dims,legend=False):
    base_path=Path(root)/exp_id
    plots_dir=base_path/"plots"
    plots_dir.mkdir(parents=True,exist_ok=True)
    data_path=base_path/"results_cumalative_sampling.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {data_path}")
    data=pd.read_csv(data_path)
    
    if len(feature_dims)==1:
        fd=feature_dims[0]
        df_fd=data[data.feature_dim==fd]
        if df_fd.empty:
            print(f"\n=== Experiment: {exp_id} ===\nKeine Daten für feature_dim={fd}\n")
            return
        fig,ax=plt.subplots(1,1,figsize=(9,7),tight_layout=True)
        plot_frd_vs_n(ax,df_fd,color_map,legend)
        ax.set_title(f"FRD vs N (feature_dim={fd})")
        if legend:
            ax.legend(ncol=2,fontsize=10)
        plot_path=plots_dir/f"frd_vs_N_featuredim_{fd}_{exp_id}.png"
        fig.savefig(plot_path,dpi=300)
        plt.close(fig)
    
    else:
        nrows=len(feature_dims)
        fig,axes=plt.subplots(nrows=nrows,ncols=1,figsize=(11,5*nrows),tight_layout=True,sharex=True)
        if nrows==1:
            axes=np.array([axes])
        for i,fd in enumerate(feature_dims):
            df_fd=data[data.feature_dim==fd]
            ax=axes[i]
            plot_frd_vs_n(ax,df_fd,color_map,legend=False)
            ax.set_title(f"FRD vs N (feature_dim={fd})")
        if legend:
            handles,labels=axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles,labels,loc="upper center",ncol=3,fontsize=10)
        plot_path=plots_dir/f"frd_vs_N_all_featuredims_{exp_id}.png"
        fig.savefig(plot_path,dpi=300)
        plt.close(fig)

    used_keys=sorted(set(key_from_dict(ast.literal_eval(cp)) for cp in data["comparison_params"].unique()))
    print(f"\n=== Experiment: {exp_id} ===")
    print("Generator → Farbe (HEX):")
    for k in used_keys:
        print(f"  {k} → {mcolors.to_hex(color_map.get(k,(0.3,0.3,0.3,1.0)))}")


def main():
    parser=argparse.ArgumentParser(description="FRD vs Sample Size; Subplots nur für Feature-Dimensionen, die in allen Experimenten vorhanden sind.")
    parser.add_argument("--root",default="./experiments/results/fd_different_parameters")
    parser.add_argument("--exp_ids","-e",nargs="+",required=True)
    parser.add_argument("--feature_dims","-d",nargs="+")
    parser.add_argument("--legend",action="store_true")
    args=parser.parse_args()
    
    requested=None if args.feature_dims is None else args.feature_dims
    all_keys=collect_color_keys_across_experiments(args.root,args.exp_ids)
    color_map=build_global_color_map(all_keys)
    feature_dims=intersect_feature_dims(args.root,args.exp_ids,requested)
    
    print("=== Globales Farb-Mapping (über alle Experimente) ===")
    for k in sorted(all_keys):
        print(f"  {k} → {mcolors.to_hex(color_map[k])}")
    if requested is None:
        print(f"Schnittmenge feature_dims: {feature_dims}\n")
    else:
        print(f"Angefragt: {sorted(set(map(int,requested)))} | Schnittmenge verfügbar: {feature_dims}\n")
    for exp_id in args.exp_ids:
        plot_experiment(args.root,exp_id,color_map,feature_dims,legend=args.legend)

if __name__=="__main__":
    main()
