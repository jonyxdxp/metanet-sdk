# ── Plot all four aspects ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('DMI Utterance Embedding Space (UMAP)', fontsize=14, y=1.02)

configs = [
    {
        'title':   'Topic Domain',
        'labels':  np.array(domains),
        'names':   {0:'Finance', 1:'Transport', 2:'Work', 3:'Health', 4:'Food'},
        'colors':  {0:'#e41a1c', 1:'#377eb8', 2:'#4daf4a',
                    3:'#984ea3', 4:'#ff7f00'},
    },
    {
        'title':   'Discourse Act',
        'labels':  np.array(acts),
        'names':   {0:'Inform', 1:'Question', 2:'Directive', 3:'Commissive'},
        'colors':  {0:'#1f77b4', 1:'#d62728', 2:'#2ca02c', 3:'#ff7f0e'},
    },
    {
        'title':   'Dialogue Phase',
        'labels':  np.array(phases),
        'names':   {0:'Opening', 1:'Middle', 2:'Closing'},
        'colors':  {0:'#2ca02c', 1:'#1f77b4', 2:'#d62728'},
    },
    {
        'title':   'Speaker Role',
        'labels':  np.array(roles),
        'names':   {0:'Speaker A', 1:'Speaker B'},
        'colors':  {0:'#e377c2', 1:'#7f7f7f'},
    },
]

for ax, cfg in zip(axes, configs):
    for label_id, name in cfg['names'].items():
        mask = cfg['labels'] == label_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=cfg['colors'][label_id],
                   label=name, s=8, alpha=0.6, linewidths=0)

    ax.set_title(cfg['title'], fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    legend = ax.legend(fontsize=8, markerscale=2,
                       loc='lower right', framealpha=0.8)

plt.tight_layout()
plt.savefig(f'{ckpt_dir}/dmi_embedding_space.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → dmi_embedding_space.png")









# :::::::::::::::::::::::::::::::::::






# ── Bonus: single-turn vs full-dialogue embeddings side by side ───────────────
# Show that single-turn embeddings cluster better than full-dialogue ones

print("Encoding full-dialogue c_t vectors for comparison...")
embs_ctx, domains_ctx = [], []
random.seed(42)

for dialog in tqdm(random.sample(dialogs, 400)):
    utts   = [u.strip() for u in dialog.strip().split('__eou__') if u.strip()]
    if len(utts) < 3: continue
    utts   = utts[:MAX_TURNS]
    domain = get_topic_domain(utts)
    if domain == -1: continue
    embs_ctx.append(encode_context(utts).numpy())
    domains_ctx.append(domain)

X_ctx  = np.array(embs_ctx)
reducer2 = umap.UMAP(n_neighbors=15, min_dist=0.1,
                     n_components=2, random_state=42, metric='cosine')
X_ctx_2d = reducer2.fit_transform(X_ctx)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {0:'#e41a1c', 1:'#377eb8', 2:'#4daf4a', 3:'#984ea3', 4:'#ff7f00'}
names  = {0:'Finance', 1:'Transport', 2:'Work', 3:'Health', 4:'Food'}

for ax, X_plot, dom_labels, title in [
    (axes[0], X_2d,     np.array(domains),     'Single-turn embeddings\n(encode_single)'),
    (axes[1], X_ctx_2d, np.array(domains_ctx), 'Full-dialogue embeddings\n(encode_context)'),
]:
    for lid, name in names.items():
        mask = dom_labels == lid
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                   c=colors[lid], label=name,
                   s=12, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, markerscale=2, loc='lower right')

fig.suptitle('DMI Embedding Space: Single-turn vs Full-dialogue (Topic Domain)',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{ckpt_dir}/dmi_single_vs_context.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → dmi_single_vs_context.png")