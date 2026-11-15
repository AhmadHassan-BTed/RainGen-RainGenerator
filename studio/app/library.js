/**
 * Library panel: drag-and-drop palette of world objects (MASTER_PLAN 7.6
 * "build the world, not the knobs") plus an accessible scene outline that
 * mirrors the viewport as a keyboard / screen-reader operable list.
 */
import { make, MATERIALS } from './presets.js';
import { uid } from './scene-store.js';

const id = (t) => uid(t);

export const LIBRARY = [
  {
    group: 'Ground & surfaces',
    items: [
      { key: 'tin-roof', label: 'Tin roof', desc: 'Thin steel sheet, bright ring (critical freq ~24 kHz)', material: 'tin_roof', make: (p) => make.surface(id('tin'), 'tin_roof', p[0], p[1], 4, 3, Math.max(p[2], 0.05)) },
      { key: 'glass', label: 'Glass', desc: 'Pane lying flat (skylight, table)', material: 'glass', make: (p) => make.surface(id('glass'), 'glass', p[0], p[1], 1.5, 1, Math.max(p[2], 0.05)) },
      { key: 'concrete', label: 'Concrete', desc: 'Hard slab: clicks and splash tails', material: 'concrete', make: (p) => make.surface(id('concrete'), 'concrete', p[0], p[1], 5, 5, p[2]) },
      { key: 'puddle', label: 'Puddle', desc: 'Shallow water: bubble plinks', material: 'puddle', make: (p) => make.surface(id('puddle'), 'puddle', p[0], p[1], 1.6, 1.1, p[2]) },
      { key: 'pond', label: 'Pond', desc: 'Deep water: Minnaert bubbles, small-drop ring', material: 'water', make: (p) => make.water(id('pond'), p[0], p[1], 6, 4, 0.8) },
      { key: 'leaves', label: 'Leaves', desc: 'Leaf litter / foliage patter', material: 'leaves', make: (p) => make.surface(id('leaves'), 'leaves', p[0], p[1], 4, 4, p[2]) },
      { key: 'gravel', label: 'Gravel', desc: 'Loose stones: soft crunchy ticks', material: 'gravel', make: (p) => make.surface(id('gravel'), 'gravel', p[0], p[1], 4, 3, p[2]) },
      { key: 'grass-patch', label: 'Grass', desc: 'Soft ground: absorbs, damps', material: 'grass', make: (p) => make.surface(id('grass'), 'grass', p[0], p[1], 5, 5, p[2]) },
    ],
  },
  {
    group: 'Structures',
    items: [
      { key: 'roof', label: 'Roof', desc: 'Gable roof (tiles)', material: 'tile', make: (p) => make.roof(id('roof'), 'tile', p[0], p[1], 2.8, 6, 5) },
      { key: 'window', label: 'Window', desc: 'Glazing: rain ticks, transmission loss', material: 'glass', make: (p) => make.window(id('window'), p[0], p[1], 1.0, 1.2, 1.2) },
      { key: 'wall', label: 'Wall', desc: 'Reflecting wall (brick)', material: 'brick', make: (p) => make.wall(id('wall'), 'brick', p[0], p[1], 6, 3) },
      { key: 'porch', label: 'Porch', desc: 'Shed roof on posts', material: 'wood', make: (p) => make.porch(id('porch'), p[0], p[1], 6, 3, 2.6, 'wood') },
      { key: 'tent', label: 'Tent', desc: 'Tensioned membrane: low dense modes', material: 'tent_fabric', make: (p) => make.tent(id('tent'), p[0], p[1]) },
      { key: 'umbrella', label: 'Umbrella', desc: 'Close membrane over the head', material: 'umbrella_fabric', make: (p) => make.umbrella(id('umbrella'), p[0], p[1]) },
      { key: 'car', label: 'Car', desc: 'Steel roof, glass, wipers', material: 'car_body', make: (p) => make.car(id('car'), p[0], p[1]) },
    ],
  },
  {
    group: 'Nature',
    items: [
      { key: 'deciduous', label: 'Deciduous tree', desc: 'Broadleaf: rustle, canopy drips', material: 'leaves', make: (p) => make.tree(id('tree'), p[0], p[1], 10, 'deciduous') },
      { key: 'conifer', label: 'Conifer', desc: 'Needle whoosh (Strouhal), drips', material: 'needles', make: (p) => make.tree(id('conifer'), p[0], p[1], 16, 'conifer') },
      { key: 'grass-field', label: 'Grass field', desc: 'Swaying field, soft ground', material: 'grass', make: (p) => make.grass(id('field'), p[0], p[1], 14, 10) },
    ],
  },
  {
    group: 'Water flow',
    items: [
      { key: 'gutter', label: 'Gutter', desc: 'Trickle + bubble cloud (hypothesis model)', material: 'metal', make: (p) => make.gutter(id('gutter'), p[0], p[1], 2.6, 5) },
      { key: 'downspout', label: 'Downspout', desc: 'Pipe resonance, gurgle', material: 'metal', make: (p) => make.downspout(id('downspout'), p[0], p[1], 2.6) },
      { key: 'eave', label: 'Eave drips', desc: 'Integrate-and-fire drips (Tate drops)', material: 'wood', make: (p) => make.eave(id('eave'), p[0], p[1], 2.6, 4) },
    ],
  },
  {
    group: 'Wind objects',
    items: [
      { key: 'wire', label: 'Wire', desc: 'Aeolian tone f = 0.2·U/d', material: 'metal', make: (p) => make.wire(id('wire'), p[0], p[1], 5, 12) },
      { key: 'chime', label: 'Wind chime', desc: 'Tubes struck by gusts', material: 'metal', make: (p) => make.chime(id('chime'), p[0], p[1], 2.2) },
      { key: 'flag', label: 'Flag', desc: 'Flapping fabric, rope slap', material: 'fabric', make: (p) => make.flag(id('flag'), p[0], p[1]) },
    ],
  },
  {
    group: 'Weather',
    items: [
      {
        key: 'storm', label: 'Storm cell', desc: 'Moving thunderstorm with a track', material: 'slate',
        make: (p, scene) => {
          const from = scene?.weather?.wind?.from_deg ?? 270;
          const to = ((from + 180) * Math.PI) / 180;
          const dx = Math.sin(to);
          const dy = Math.cos(to);
          const c = [p[0], p[1]];
          const track = [
            [Math.round(c[0] - dx * 5000), Math.round(c[1] - dy * 5000)],
            [Math.round(c[0]), Math.round(c[1])],
            [Math.round(c[0] + dx * 5000), Math.round(c[1] + dy * 5000)],
          ];
          return make.storm(id('storm'), track, 25, 3, 0.6, 4);
        },
      },
    ],
  },
];

export const TEMPLATE_BY_KEY = new Map(LIBRARY.flatMap((g) => g.items.map((i) => [i.key, i])));

/** Create a scene object from a library template at a world position [x, y, z]. */
export function instantiate(key, position, scene) {
  const t = TEMPLATE_BY_KEY.get(key);
  if (!t) return null;
  const obj = t.make(position || [0, 0, 0], scene);
  obj.label = t.label;
  return obj;
}

const TYPE_LABEL = {
  surface: 'Surface', water: 'Water', roof: 'Roof', window: 'Window', wall: 'Wall', tree: 'Tree', grass: 'Grass',
  gutter: 'Gutter', downspout: 'Downspout', eave: 'Eave', wire: 'Wire', chime: 'Chime', flag: 'Flag', car: 'Car',
  tent: 'Tent', umbrella: 'Umbrella', storm: 'Storm cell',
};

export function objectLabel(o) {
  if (o.label) return o.label;
  const mat = MATERIALS[o.material]?.label;
  const t = TYPE_LABEL[o.type] || o.type;
  return mat && o.type === 'surface' ? mat : t;
}

export class Library {
  /**
   * @param {HTMLElement} root
   * @param {{store: import('./scene-store.js').SceneStore, onAdd: (key: string) => void}} opts
   */
  constructor(root, { store, onAdd }) {
    this.root = root;
    this.store = store;
    this.onAdd = onAdd;
    this._build();
    store.on('change', (e) => {
      if (e.kind === 'load' || e.kind === 'restore' || e.kind === 'object-add' || e.kind === 'object-remove' || (e.kind === 'object' && e.paths.some((p) => p.endsWith('label')))) this.renderOutline();
    });
    store.on('select', () => this._markSelection());
    this.renderOutline();
  }

  _build() {
    const root = this.root;
    root.textContent = '';
    const head = document.createElement('div');
    head.className = 'panel-head';
    head.innerHTML = '<h2 id="library-title">Library</h2>';
    const search = document.createElement('input');
    search.type = 'search';
    search.placeholder = 'Filter objects…';
    search.className = 'lib-search';
    search.setAttribute('aria-label', 'Filter library objects');
    head.appendChild(search);
    root.appendChild(head);

    const hint = document.createElement('p');
    hint.className = 'hint';
    hint.textContent = 'Drag into the viewport, or press Enter to place in front of the listener.';
    root.appendChild(hint);

    const groups = document.createElement('div');
    groups.className = 'lib-groups';
    for (const g of LIBRARY) {
      const det = document.createElement('details');
      det.open = true;
      det.className = 'lib-group';
      const sum = document.createElement('summary');
      sum.textContent = g.group;
      det.appendChild(sum);
      const list = document.createElement('ul');
      list.className = 'lib-list';
      list.setAttribute('aria-label', g.group);
      for (const it of g.items) {
        const li = document.createElement('li');
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'lib-item';
        b.draggable = true;
        b.dataset.key = it.key;
        b.setAttribute('aria-label', `Add ${it.label}: ${it.desc}`);
        const sw = document.createElement('span');
        sw.className = `swatch swatch-${it.key}`;
        sw.style.setProperty('--swatch', MATERIALS[it.material]?.color || '#777');
        sw.setAttribute('aria-hidden', 'true');
        const txt = document.createElement('span');
        txt.className = 'lib-text';
        const nm = document.createElement('span');
        nm.className = 'lib-name';
        nm.textContent = it.label;
        const ds = document.createElement('span');
        ds.className = 'lib-desc';
        ds.textContent = it.desc;
        txt.append(nm, ds);
        b.append(sw, txt);
        b.addEventListener('click', () => this.onAdd?.(it.key));
        b.addEventListener('dragstart', (e) => {
          e.dataTransfer.setData('application/x-raingen-template', it.key);
          e.dataTransfer.setData('text/plain', `raingen:${it.key}`);
          e.dataTransfer.effectAllowed = 'copy';
          document.body.classList.add('dragging-template');
        });
        b.addEventListener('dragend', () => document.body.classList.remove('dragging-template'));
        li.appendChild(b);
        list.appendChild(li);
      }
      det.appendChild(list);
      groups.appendChild(det);
    }
    root.appendChild(groups);

    search.addEventListener('input', () => {
      const q = search.value.trim().toLowerCase();
      groups.querySelectorAll('.lib-item').forEach((b) => {
        const t = TEMPLATE_BY_KEY.get(b.dataset.key);
        const hit = !q || `${t.label} ${t.desc}`.toLowerCase().includes(q);
        b.parentElement.hidden = !hit;
      });
      groups.querySelectorAll('.lib-group').forEach((d) => {
        d.hidden = ![...d.querySelectorAll('li')].some((li) => !li.hidden);
        if (q) d.open = true;
      });
    });

    const outlineWrap = document.createElement('section');
    outlineWrap.className = 'outline';
    outlineWrap.setAttribute('aria-labelledby', 'outline-title');
    outlineWrap.innerHTML = '<h3 id="outline-title">Scene outline</h3>';
    this.outline = document.createElement('ul');
    this.outline.className = 'outline-list';
    this.outline.setAttribute('role', 'listbox');
    this.outline.setAttribute('aria-labelledby', 'outline-title');
    this.outline.tabIndex = 0;
    outlineWrap.appendChild(this.outline);
    root.appendChild(outlineWrap);

    this.outline.addEventListener('keydown', (e) => {
      const items = [...this.outline.querySelectorAll('[role="option"]')];
      if (!items.length) return;
      let idx = items.findIndex((x) => x.getAttribute('aria-selected') === 'true');
      if (e.key === 'ArrowDown') idx = Math.min(items.length - 1, idx + 1);
      else if (e.key === 'ArrowUp') idx = Math.max(0, idx - 1);
      else if (e.key === 'Home') idx = 0;
      else if (e.key === 'End') idx = items.length - 1;
      else if ((e.key === 'Delete' || e.key === 'Backspace') && this.store.selection) {
        e.preventDefault();
        e.stopPropagation();
        this.store.removeObject(this.store.selection);
        return;
      } else return;
      e.preventDefault();
      e.stopPropagation();
      const target = items[Math.max(0, idx)];
      this.store.select(target.dataset.id || null);
      target.scrollIntoView({ block: 'nearest' });
    });
  }

  renderOutline() {
    const ul = this.outline;
    if (!ul) return;
    ul.textContent = '';
    const objs = this.store.scene.place.objects;
    if (!objs.length) {
      const li = document.createElement('li');
      li.className = 'outline-empty';
      li.textContent = 'No objects yet. Add some from the library.';
      ul.appendChild(li);
      return;
    }
    for (const o of objs) {
      const li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.dataset.id = o.id;
      li.className = `outline-item type-${o.type}`;
      const sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.setProperty('--swatch', MATERIALS[o.material]?.color || (o.type === 'storm' ? '#5b6270' : '#777'));
      sw.setAttribute('aria-hidden', 'true');
      const nm = document.createElement('span');
      nm.className = 'outline-name';
      nm.textContent = objectLabel(o);
      const idEl = document.createElement('span');
      idEl.className = 'outline-id';
      idEl.textContent = o.id;
      li.append(sw, nm, idEl);
      li.addEventListener('click', () => this.store.select(o.id));
      ul.appendChild(li);
    }
    this._markSelection();
  }

  _markSelection() {
    const sel = this.store.selection;
    this.outline?.querySelectorAll('[role="option"]').forEach((li) => {
      const on = li.dataset.id === sel;
      li.setAttribute('aria-selected', on ? 'true' : 'false');
      if (on) {
        this.outline.setAttribute('aria-activedescendant', li.id || (li.id = `outline-${sel}`));
      }
    });
  }
}
