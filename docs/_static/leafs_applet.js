/* LEAFS - interactive applet for the pyLEAFS documentation.
 *
 * The model is a direct port of the library's 2d world, the one
 * Simulation.forager builds: the resource field steps as
 * fields.ResourceField.step does, region by region, and the foragers as
 * population.Population.step does, stage by stage and in the same order.
 * Every random number the model uses passes through one function, draw,
 * so the port can be checked against the Python by replaying the
 * Python's own draws into it. No dependencies, no build step.
 */
(function () {
    'use strict';

    /* ── The world of Simulation.forager ────────────────────────────────── */
    const NX = 10, NY = 10;             // regions per axis
    const L = 10;                       // region side
    const NREG = NX * NY;
    const EXT_X = NX * L, EXT_Y = NY * L;
    const DT = 0.02;
    const GAMMA = 0.001;                // resource birth rate per unit volume
    const DECAY = 0.1;                  // resource decay rate
    const R_COL = 1.0;                  // collection radius
    const MU0 = 0.1;                    // basal metabolic rate
    const S_MAX = 1.0;                  // maximum fuel
    const REPRO = 0.8 * S_MAX;          // budding threshold
    const TAU_A = 1.0;                  // heading persistence time
    const N_AGENTS = 5;                 // initial foragers
    const N_A_MAX = GAMMA * EXT_X * EXT_Y / MU0;   // agent carrying capacity

    /* Moore offsets in the order Grid.offsets lists them, centre first:
     * itertools.product((0, -1, 1), repeat=2). */
    const OFFSETS = [[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1],
                     [-1, 1], [1, 0], [1, -1], [1, 1]];

    /* ── Arithmetic as NumPy does it ────────────────────────────────────── */

    /* np.mod: the remainder takes the divisor's sign, and a tiny negative
     * input can land exactly on the divisor. */
    function pymod(a, b) {
        let m = a % b;
        if (m !== 0 && (m < 0) !== (b < 0)) m += b;
        return m;
    }

    /* np.round: halves go to the even neighbour. */
    function rint(x) {
        const r = Math.round(x);
        return (r - x === 0.5 && r % 2 !== 0) ? r - 1 : r;
    }

    /* Grid.displacement, one axis: the minimum image of b - a. */
    function minImage(d, ext) {
        return d - ext * rint(d / ext);
    }

    /* ── Random numbers ─────────────────────────────────────────────────── */

    /* The one door every random number comes through. kind is 'uniform'
     * (rng.random), 'normal' (rng.standard_normal), 'poisson' (rng.poisson),
     * 'binomial' (rng.binomial), or 'choice', which returns k distinct
     * integers below n (rng.choice(n, size=k, replace=False)). */
    let gaussSpare = null;
    function draw(kind, a, b) {
        switch (kind) {
        case 'uniform': return Math.random();
        case 'normal': return gauss();
        case 'poisson': return poisson(a);
        case 'binomial': return binomial(a, b);
        case 'choice': return choice(a, b);
        }
        throw new Error('unknown draw ' + kind);
    }

    /* Polar Box-Muller standard normal, one spare cached. */
    function gauss() {
        if (gaussSpare !== null) {
            const g = gaussSpare; gaussSpare = null; return g;
        }
        let u, v, s;
        do {
            u = 2 * Math.random() - 1;
            v = 2 * Math.random() - 1;
            s = u * u + v * v;
        } while (s >= 1 || s === 0);
        const m = Math.sqrt(-2 * Math.log(s) / s);
        gaussSpare = v * m;
        return u * m;
    }

    /* log Gamma(x) by the Stirling series NumPy's samplers use. */
    const LOGGAM_A = [8.333333333333333e-02, -2.777777777777778e-03,
                      7.936507936507937e-04, -5.952380952380952e-04,
                      8.417508417508418e-04, -1.917526917526918e-03,
                      6.410256410256410e-03, -2.955065359477124e-02,
                      1.796443723688307e-01, -1.39243221690590e+00];
    function loggam(x) {
        if (x === 1 || x === 2) return 0;
        const n = x < 7 ? Math.floor(7 - x) : 0;
        let x0 = x + n;
        const x2 = (1 / x0) * (1 / x0);
        let gl0 = LOGGAM_A[9];
        for (let k = 8; k >= 0; k--) gl0 = gl0 * x2 + LOGGAM_A[k];
        let gl = gl0 / x0 + 0.9189385332046727 + (x0 - 0.5) * Math.log(x0) - x0;
        for (let k = 1; k <= n; k++) { gl -= Math.log(x0 - 1); x0 -= 1; }
        return gl;
    }

    /* Poisson: multiplication of uniforms for a small mean, Hormann's
     * transformed rejection (PTRS) for a large one, as NumPy splits it. */
    function poisson(lam) {
        if (lam <= 0) return 0;
        if (lam < 10) {
            const enlam = Math.exp(-lam);
            let k = 0, prod = 1;
            for (;;) {
                prod *= Math.random();
                if (prod <= enlam) return k;
                k++;
            }
        }
        const slam = Math.sqrt(lam), loglam = Math.log(lam);
        const bb = 0.931 + 2.53 * slam, aa = -0.059 + 0.02483 * bb;
        const invalpha = 1.1239 + 1.1328 / (bb - 3.4);
        const vr = 0.9277 - 3.6224 / (bb - 2);
        for (;;) {
            const U = Math.random() - 0.5, V = Math.random();
            const us = 0.5 - Math.abs(U);
            const k = Math.floor((2 * aa / us + bb) * U + lam + 0.43);
            if (us >= 0.07 && V <= vr) return k;
            if (k < 0 || (us < 0.013 && V > us)) continue;
            if (Math.log(V) + Math.log(invalpha) - Math.log(aa / (us * us) + bb)
                <= -lam + k * loglam - loggam(k + 1)) return k;
        }
    }

    /* Binomial by inversion, the method NumPy uses whenever n p < 30;
     * past that, which this world never reaches, count the coin flips. */
    function binomial(n, p) {
        if (n <= 0 || p <= 0) return 0;
        if (p >= 1) return n;
        if (p > 0.5) return n - binomial(n, 1 - p);
        if (n * p >= 30) {
            let k = 0;
            for (let i = 0; i < n; i++) if (Math.random() < p) k++;
            return k;
        }
        const q = 1 - p, qn = Math.exp(n * Math.log(q)), np = n * p;
        const bound = Math.min(n, np + 10 * Math.sqrt(np * q + 1));
        let x = 0, px = qn, u = Math.random();
        while (u > px) {
            x++;
            if (x > bound) {
                x = 0; px = qn; u = Math.random();
            } else {
                u -= px;
                px = ((n - x + 1) * p * px) / (x * q);
            }
        }
        return x;
    }

    /* k distinct integers below n, by Floyd's algorithm. */
    function choice(n, k) {
        const picked = new Set();
        for (let j = n - k; j < n; j++) {
            const t = Math.floor(Math.random() * (j + 1));
            picked.add(picked.has(t) ? j : t);
        }
        return Array.from(picked);
    }

    /* ── The model (ports grid.py, fields.py, population.py) ────────────── */

    /* One world: a ResourceField and a greedy Population on the shared
     * 10 x 10 grid. opts carries Xi, R_sense, v, sigma and, for the check
     * against the Python, a replacement for draw. */
    function World(opts) {
        opts = opts || {};
        this.draw = opts.draw || draw;
        this.Rsense = opts.R_sense !== undefined ? opts.R_sense : 6.0;
        this.v = opts.v !== undefined ? opts.v : 20.0;
        this.sigma = opts.sigma !== undefined ? opts.sigma : 0.1;
        this.steps = 0;

        // ResourceField: region r owns slots [r * stride, r * stride + cap)
        // of a flat buffer of positions normalised to [0, 1)^2.
        this.count = new Int32Array(NREG);
        this.stride = 0;
        this.lx = null;
        this.ly = null;
        this.setXi(opts.Xi !== undefined ? opts.Xi : 0.5);

        // Population: parallel arrays, grown by doubling.
        this.n = 0;
        this.nextId = 0;
        this.room = 0;
        this.grow(256);

        this.seedResources();
        this.seedAgents(opts.nAgents !== undefined ? opts.nAgents : N_AGENTS);
    }

    /* Simulation.forager's energy per resource and ResourceField's derived
     * rates: epsilon = Gamma / ((Xi / r_col)^2 gamma). */
    World.prototype.setXi = function (Xi) {
        this.Xi = Xi;
        this.epsilon = GAMMA / ((Xi / R_COL) ** 2 * DECAY);
        this.Neq = GAMMA * L ** 2 / (this.epsilon * DECAY);
        this.lam = GAMMA * L ** 2 * DT / this.epsilon;
        this.pDie = 1.0 - Math.exp(-DECAY * DT);
        this.cap = Math.max(20, Math.ceil(4.0 * this.Neq));
        if (this.cap > this.stride) {
            const stride = this.cap;
            const lx = new Float64Array(NREG * stride);
            const ly = new Float64Array(NREG * stride);
            for (let r = 0; r < NREG; r++) {
                for (let s = 0; s < this.count[r]; s++) {
                    lx[r * stride + s] = this.lx[r * this.stride + s];
                    ly[r * stride + s] = this.ly[r * this.stride + s];
                }
            }
            this.stride = stride;
            this.lx = lx;
            this.ly = ly;
        }
    };

    /* ResourceField.seed: Poisson(N_eq) per region, all counts first. */
    World.prototype.seedResources = function () {
        const counts = new Array(NREG);
        for (let r = 0; r < NREG; r++) counts[r] = this.draw('poisson', this.Neq);
        this.count.fill(0);
        for (let r = 0; r < NREG; r++) {
            this.growRegion(r, Math.min(counts[r], this.cap));
        }
    };

    /* ResourceField._grow_region: k uniform positions, up to capacity. */
    World.prototype.growRegion = function (r, k) {
        const n = this.count[r];
        k = Math.min(k, this.cap - n);
        if (k <= 0) return;
        const base = r * this.stride + n;
        for (let i = 0; i < k; i++) {
            this.lx[base + i] = this.draw('uniform');
            this.ly[base + i] = this.draw('uniform');
        }
        this.count[r] = n + k;
    };

    /* ResourceField._decay_region: drop k distinct slots, keeping the
     * survivors in order. */
    World.prototype.decayRegion = function (r, k) {
        const n = this.count[r];
        if (n === 0 || k <= 0) return;
        k = Math.min(k, n);
        const drop = this.draw('choice', n, k);
        const keep = new Uint8Array(n).fill(1);
        for (let i = 0; i < drop.length; i++) keep[drop[i]] = 0;
        const base = r * this.stride;
        let m = 0;
        for (let s = 0; s < n; s++) {
            if (keep[s]) {
                this.lx[base + m] = this.lx[base + s];
                this.ly[base + m] = this.ly[base + s];
                m++;
            }
        }
        this.count[r] = n - k;
    };

    /* ResourceField.step: every region's births, then every region's
     * deaths, then every region's coin for which comes first. */
    World.prototype.stepResources = function () {
        const kGrow = new Array(NREG), kDecay = new Array(NREG);
        const growFirst = new Array(NREG);
        for (let r = 0; r < NREG; r++) kGrow[r] = this.draw('poisson', this.lam);
        for (let r = 0; r < NREG; r++) kDecay[r] = this.draw('binomial', this.count[r], this.pDie);
        for (let r = 0; r < NREG; r++) growFirst[r] = this.draw('uniform') > 0.5;
        for (let r = 0; r < NREG; r++) {
            if (growFirst[r]) {
                this.growRegion(r, kGrow[r]);
                this.decayRegion(r, kDecay[r]);
            } else {
                this.decayRegion(r, kDecay[r]);
                this.growRegion(r, kGrow[r]);
            }
        }
    };

    /* ResourceField.harvest: swap-delete one slot; false if it is gone. */
    World.prototype.harvestSlot = function (r, slot) {
        const n = this.count[r];
        if (!(slot >= 0 && slot < n)) return false;
        const base = r * this.stride;
        this.lx[base + slot] = this.lx[base + n - 1];
        this.ly[base + slot] = this.ly[base + n - 1];
        this.count[r] = n - 1;
        return true;
    };

    /* ResourceField.neighbourhood: displacement, region and slot of every
     * resource in the Moore neighbourhood of (x, y), centre region first. */
    World.prototype.neighbourhood = function (x, y) {
        const ix = pymod(Math.floor(pymod(x, EXT_X) / L), NX);
        const iy = pymod(Math.floor(pymod(y, EXT_Y) / L), NY);
        const out = [];
        for (let o = 0; o < OFFSETS.length; o++) {
            const jx = pymod(ix + OFFSETS[o][0], NX);
            const jy = pymod(iy + OFFSETS[o][1], NY);
            const r = jx * NY + jy;
            const n = this.count[r], base = r * this.stride;
            const cx = jx * L, cy = jy * L;
            for (let s = 0; s < n; s++) {
                const dx = minImage(cx + this.lx[base + s] * L - x, EXT_X);
                const dy = minImage(cy + this.ly[base + s] * L - y, EXT_Y);
                out.push(dx, dy, r, s);
            }
        }
        return out;
    };

    World.prototype.total = function () {
        let t = 0;
        for (let r = 0; r < NREG; r++) t += this.count[r];
        return t;
    };

    /* Population._ensure_capacity. */
    World.prototype.grow = function (need) {
        if (need <= this.room) return;
        const room = Math.max(need, 2 * this.room);
        const keep = (A, T) => {
            const B = new T(room);
            if (A) B.set(A.subarray(0, this.n));
            return B;
        };
        this.px = keep(this.px, Float64Array);
        this.py = keep(this.py, Float64Array);
        this.hx = keep(this.hx, Float64Array);
        this.hy = keep(this.hy, Float64Array);
        this.fuel = keep(this.fuel, Float64Array);
        this.age = keep(this.age, Int32Array);
        this.ids = keep(this.ids, Int32Array);
        this.harvested = keep(this.harvested, Int32Array);
        this.offspring = keep(this.offspring, Int32Array);
        this.room = room;
    };

    /* Population._random_headings, for one agent: a normalised Gaussian. */
    World.prototype.randomHeading = function () {
        const vx = this.draw('normal'), vy = this.draw('normal');
        let norm = Math.sqrt(vx * vx + vy * vy);
        if (norm === 0) norm = 1.0;
        return [vx / norm, vy / norm];
    };

    /* Population.add: one agent, wrapped into the box; a given heading is
     * normalised, a missing one drawn; fuel defaults to s_max. */
    World.prototype.add = function (x, y, heading, fuel) {
        this.grow(this.n + 1);
        const i = this.n;
        this.px[i] = pymod(x, EXT_X);
        this.py[i] = pymod(y, EXT_Y);
        if (heading === undefined) {
            heading = this.randomHeading();
        } else {
            const norm = Math.sqrt(heading[0] * heading[0] + heading[1] * heading[1]);
            heading = norm > 0 ? [heading[0] / norm, heading[1] / norm]
                               : this.randomHeading();
        }
        this.hx[i] = heading[0];
        this.hy[i] = heading[1];
        this.fuel[i] = fuel === undefined ? S_MAX : fuel;
        this.age[i] = 0;
        this.ids[i] = this.nextId++;
        this.harvested[i] = 0;
        this.offspring[i] = 0;
        this.n++;
        return this.ids[i];
    };

    /* Population.seed: every position, then every heading, then add. */
    World.prototype.seedAgents = function (n) {
        const pos = [], head = [];
        for (let k = 0; k < n; k++) {
            pos.push([this.draw('uniform') * EXT_X, this.draw('uniform') * EXT_Y]);
        }
        for (let k = 0; k < n; k++) head.push(this.randomHeading());
        for (let k = 0; k < n; k++) this.add(pos[k][0], pos[k][1], head[k]);
    };

    World.prototype.indexOf = function (id) {
        for (let i = 0; i < this.n; i++) if (this.ids[i] === id) return i;
        return -1;
    };

    /* Population.step: steer, diffuse, move, harvest, metabolise, bud,
     * cull, each stage over every agent before the next begins. */
    World.prototype.stepAgents = function () {
        const n = this.n;
        if (n === 0) return;

        // Greedy steering, from ResourceField.sample: the nearest resource
        // in the Moore neighbourhood, if it lies within the sensing radius.
        const R2 = this.Rsense ** 2;
        const aimX = new Float64Array(n), aimY = new Float64Array(n);
        const aim = new Uint8Array(n);
        for (let i = 0; i < n; i++) {
            const nb = this.neighbourhood(this.px[i], this.py[i]);
            let best = Infinity, bx = 0, by = 0;
            for (let j = 0; j < nb.length; j += 4) {
                const d2 = nb[j] * nb[j] + nb[j + 1] * nb[j + 1];
                if (d2 < best) { best = d2; bx = nb[j]; by = nb[j + 1]; }
            }
            if (best <= R2) { aim[i] = 1; aimX[i] = bx; aimY[i] = by; }
        }
        for (let i = 0; i < n; i++) {
            if (!aim[i]) continue;
            let norm = Math.sqrt(aimX[i] * aimX[i] + aimY[i] * aimY[i]);
            if (norm === 0) norm = 1.0;
            this.hx[i] = aimX[i] / norm;
            this.hy[i] = aimY[i] / norm;
        }

        // Rotational diffusion of every heading.
        const scale = this.sigma / Math.sqrt(TAU_A) * Math.sqrt(DT);
        const noise = new Float64Array(n);
        for (let i = 0; i < n; i++) noise[i] = this.draw('normal');
        for (let i = 0; i < n; i++) {
            const theta = Math.atan2(this.hy[i], this.hx[i]) + scale * noise[i];
            this.hx[i] = Math.cos(theta);
            this.hy[i] = Math.sin(theta);
        }

        // Ballistic step on the torus.
        for (let i = 0; i < n; i++) {
            this.px[i] = pymod(this.px[i] + this.v * this.hx[i] * DT, EXT_X);
            this.py[i] = pymod(this.py[i] + this.v * this.hy[i] * DT, EXT_Y);
            this.age[i]++;
        }

        // Harvest, agent by agent: every resource within r_col of it, read
        // off before it eats and removed by region and slot, each hit once,
        // highest slot first within its region, so that a swap-delete never
        // moves a resource still waiting to be taken.
        const r2 = R_COL ** 2;
        const energy = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const nb = this.neighbourhood(this.px[i], this.py[i]);
            const hits = [];
            for (let j = 0; j < nb.length; j += 4) {
                if (nb[j] * nb[j] + nb[j + 1] * nb[j + 1] > r2) continue;
                const r = nb[j + 2], s = nb[j + 3];
                if (!hits.some(h => h[0] === r && h[1] === s)) hits.push([r, s]);
            }
            hits.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
            for (const [r, s] of hits) {
                if (this.harvestSlot(r, s)) {
                    energy[i] += this.epsilon;
                    this.harvested[i]++;
                }
            }
        }

        // Metabolism, capped at s_max.
        for (let i = 0; i < n; i++) {
            this.fuel[i] = Math.min(S_MAX, this.fuel[i] - MU0 * DT + energy[i]);
        }

        // Budding: every agent at the threshold splits its fuel with a
        // daughter r_col away along a random direction, which is also the
        // daughter's heading. Daughters wait for the next step.
        const ready = [];
        for (let i = 0; i < n; i++) if (this.fuel[i] >= REPRO) ready.push(i);
        for (let k = 0; k < ready.length; k++) {
            const i = ready[k];
            const half = this.fuel[i] / 2.0;
            this.fuel[i] = half;
            this.offspring[i]++;
            const dir = this.randomHeading();
            this.add(pymod(this.px[i] + R_COL * dir[0], EXT_X),
                     pymod(this.py[i] + R_COL * dir[1], EXT_Y), dir, half);
        }

        // Death: drop every agent out of fuel, keeping the rest in order.
        let m = 0;
        for (let i = 0; i < this.n; i++) {
            if (!(this.fuel[i] > 0)) continue;
            if (m !== i) {
                this.px[m] = this.px[i]; this.py[m] = this.py[i];
                this.hx[m] = this.hx[i]; this.hy[m] = this.hy[i];
                this.fuel[m] = this.fuel[i]; this.age[m] = this.age[i];
                this.ids[m] = this.ids[i];
                this.harvested[m] = this.harvested[i];
                this.offspring[m] = this.offspring[i];
            }
            m++;
        }
        this.n = m;
    };

    /* Simulation.step: the field, then the population. */
    World.prototype.step = function () {
        this.stepResources();
        this.stepAgents();
        this.steps++;
    };

    /* Under node there is no page to draw on: hand the model to the
     * caller instead, which is how the port is checked. */
    if (typeof document === 'undefined') {
        if (typeof module !== 'undefined') {
            module.exports = { World: World, draw: draw, N_A_MAX: N_A_MAX };
        }
        return;
    }

    /* ── Scenery ────────────────────────────────────────────────────────── */
    const STEPS = 4;                // steps per frame, the Viewer's default
    const TRACE = 900;              // history length of the meters
    const SELECT_R = 2.0;           // the Viewer's select_radius
    const SELECT_PX = 14;           // ...but never less than a fingertip
    const TRAIL = 200;              // the Viewer's trail_length
    const LOG_LO = 0.01, LOG_HI = 3;    // phase plane range, both axes

    /* pyLEAFS.palette. */
    const STAGE = '#1C2030';        // wire_ink
    const BOX = '#2B3A52';          // abyss_blue
    const EDGE = '#808898';         // sensor_slate
    const FOOD = '#50C870';         // food_green
    const AGENT = '#F0A0C0';        // brain_pink
    const WINE = '#7A1838';         // motor_wine
    const MIST = '#D8E8F0';         // sensor_mist

    const P = { Xi: 0.5, R_sense: 6.0, v: 20.0, sigma: 0.1 };
    let world = null;
    let running = true;
    let selected = null;            // stable id of the followed forager
    let trail = [];
    const histA = [], histR = [];   // N_A / N_A^max and N_R / N_R^max

    let canvas, ctx, traceCv, phaseCv;
    let cssW = 0, cssH = 0;         // canvas size in CSS pixels
    let ox = 0, oy = 0, scale = 1;  // the world's box on the canvas

    function newWorld() {
        world = new World({ Xi: P.Xi, R_sense: P.R_sense, v: P.v, sigma: P.sigma });
        selected = null;
        trail = [];
        histA.length = 0;
        histR.length = 0;
        record();
    }

    function record() {
        histA.push(world.n / N_A_MAX);
        histR.push(world.total() / (world.Neq * NREG));
        if (histA.length > TRACE) { histA.shift(); histR.shift(); }
    }

    /* Physical coordinates to canvas pixels, y up as in the Viewer. */
    function sx(x) { return ox + x * scale; }
    function sy(y) { return oy + (EXT_Y - y) * scale; }

    /* Size a canvas's backing store to its CSS box at the screen's pixel
     * ratio, so the drawing is crisp on a phone. */
    function fit(cv) {
        const dpr = window.devicePixelRatio || 1;
        const w = cv.clientWidth, h = cv.clientHeight;
        const W = Math.round(w * dpr), H = Math.round(h * dpr);
        if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
        const c = cv.getContext('2d');
        c.setTransform(cv.width / w, 0, 0, cv.height / h, 0, 0);
        return c;
    }

    function layout() {
        ctx = fit(canvas);
        cssW = canvas.clientWidth;
        cssH = canvas.clientHeight;
        const side = Math.min(cssW, cssH) - 16;
        scale = side / EXT_X;
        ox = (cssW - side) / 2;
        oy = (cssH - side) / 2;
    }

    function paint() {
        const side = EXT_X * scale;
        ctx.fillStyle = STAGE;
        ctx.fillRect(0, 0, cssW, cssH);
        ctx.fillStyle = BOX;
        ctx.fillRect(ox, oy, side, side);

        ctx.save();
        ctx.beginPath();
        ctx.rect(ox, oy, side, side);
        ctx.clip();

        // Resources: dots while they are few, squares once they are many.
        ctx.fillStyle = FOOD;
        const rr = Math.max(1.1, 0.3 * scale);
        const many = world.total() > 6000;
        if (!many) ctx.beginPath();
        for (let r = 0; r < NREG; r++) {
            const cx = Math.floor(r / NY) * L, cy = (r % NY) * L;
            const base = r * world.stride;
            for (let s = 0; s < world.count[r]; s++) {
                const x = sx(cx + world.lx[base + s] * L);
                const y = sy(cy + world.ly[base + s] * L);
                if (many) {
                    ctx.fillRect(x - rr, y - rr, 2 * rr, 2 * rr);
                } else {
                    ctx.moveTo(x + rr, y);
                    ctx.arc(x, y, rr, 0, 6.283);
                }
            }
        }
        if (!many) ctx.fill();

        // The followed forager: its sensing radius and its recent path.
        const i = selected === null ? -1 : world.indexOf(selected);
        if (selected !== null && i < 0) { selected = null; trail = []; }
        if (i >= 0) {
            ctx.strokeStyle = 'rgba(216, 232, 240, 0.35)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(sx(world.px[i]), sy(world.py[i]), world.Rsense * scale, 0, 6.283);
            ctx.stroke();
            ctx.strokeStyle = AGENT;
            ctx.globalAlpha = 0.9;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let k = 0; k < trail.length; k++) {
                const [x, y] = trail[k];
                const jump = k > 0 && (Math.abs(x - trail[k - 1][0]) > EXT_X / 2
                                       || Math.abs(y - trail[k - 1][1]) > EXT_Y / 2);
                if (k === 0 || jump) ctx.moveTo(sx(x), sy(y));
                else ctx.lineTo(sx(x), sy(y));
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // Foragers, sized by their fuel as in the Viewer, with a tick
        // along the heading.
        for (let j = 0; j < world.n; j++) {
            const frac = Math.min(1, Math.max(0.1, world.fuel[j] / S_MAX));
            const rad = scale * (0.45 + 0.75 * Math.sqrt(frac));
            const x = sx(world.px[j]), y = sy(world.py[j]);
            ctx.fillStyle = AGENT;
            ctx.strokeStyle = WINE;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, rad, 0, 6.283);
            ctx.fill();
            ctx.stroke();
            ctx.strokeStyle = MIST;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + world.hx[j] * rad * 1.7, y - world.hy[j] * rad * 1.7);
            ctx.stroke();
        }
        if (i >= 0) {
            ctx.strokeStyle = MIST;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx(world.px[i]), sy(world.py[i]), scale * 1.2 + 5, 0, 6.283);
            ctx.stroke();
        }
        ctx.restore();

        ctx.strokeStyle = EDGE;
        ctx.lineWidth = 1;
        ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
    }

    /* Both populations against their carrying capacities, on [0, 2]. */
    function drawTrace() {
        const c = fit(traceCv);
        const W = traceCv.clientWidth, H = traceCv.clientHeight;
        c.clearRect(0, 0, W, H);
        c.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        c.lineWidth = 1;
        c.strokeRect(0.5, 0.5, W - 1, H - 1);
        c.setLineDash([3, 3]);
        c.beginPath();
        c.moveTo(1, H / 2);
        c.lineTo(W - 1, H / 2);
        c.stroke();
        c.setLineDash([]);
        for (const [hist, col] of [[histR, FOOD], [histA, AGENT]]) {
            c.strokeStyle = col;
            c.lineWidth = 1.4;
            c.beginPath();
            for (let k = 0; k < hist.length; k++) {
                const x = 2 + (k / (TRACE - 1)) * (W - 4);
                const y = H - 2 - Math.min(2, hist[k]) / 2 * (H - 4);
                k ? c.lineTo(x, y) : c.moveTo(x, y);
            }
            c.stroke();
        }
    }

    /* The phase plane on log axes: the census against the equilibrium
     * line N_R / N_R^max + N_A / N_A^max = 1, with the recent path fading
     * behind the present. */
    function logPos(v, len) {
        const u = Math.log(Math.min(LOG_HI, Math.max(LOG_LO, v)) / LOG_LO)
                / Math.log(LOG_HI / LOG_LO);
        return 2 + u * (len - 4);
    }

    function drawPhase() {
        const c = fit(phaseCv);
        const W = phaseCv.clientWidth, H = phaseCv.clientHeight;
        const px = v => logPos(v, W), py = v => H - logPos(v, H);
        c.clearRect(0, 0, W, H);
        c.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        c.lineWidth = 1;
        c.strokeRect(0.5, 0.5, W - 1, H - 1);
        c.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        c.beginPath();
        for (const t of [0.1, 1]) {
            c.moveTo(px(t), 1); c.lineTo(px(t), H - 1);
            c.moveTo(1, py(t)); c.lineTo(W - 1, py(t));
        }
        c.stroke();
        c.strokeStyle = 'rgba(216, 232, 240, 0.7)';
        c.setLineDash([4, 3]);
        c.beginPath();
        for (let k = 0; k <= 200; k++) {
            const x = LOG_LO * Math.pow(1 / LOG_LO, k / 200);
            const y = 1 - x;
            if (y < LOG_LO) break;
            k ? c.lineTo(px(x), py(y)) : c.moveTo(px(x), py(y));
        }
        c.stroke();
        c.setLineDash([]);
        const n = histA.length;
        c.lineWidth = 1.4;
        for (let k = 1; k < n; k++) {
            c.strokeStyle = 'rgba(240, 160, 192, ' + (0.08 + 0.9 * k / n).toFixed(3) + ')';
            c.beginPath();
            c.moveTo(px(histR[k - 1]), py(histA[k - 1]));
            c.lineTo(px(histR[k]), py(histA[k]));
            c.stroke();
        }
        if (n) {
            c.fillStyle = AGENT;
            c.beginPath();
            c.arc(px(histR[n - 1]), py(histA[n - 1]), 3, 0, 6.283);
            c.fill();
        }
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el.textContent !== text) el.textContent = text;
    }

    function readout() {
        setText('leafs-t', (world.steps * DT).toFixed(2));
        setText('leafs-na', world.n ? String(world.n) : '0, extinct');
        setText('leafs-nr', String(world.total()));
        const i = selected === null ? -1 : world.indexOf(selected);
        document.getElementById('leafs-agent').classList.toggle('on', i >= 0);
        if (i < 0) return;
        setText('leafs-id', String(world.ids[i]));
        setText('leafs-fuel', world.fuel[i].toFixed(3));
        setText('leafs-age', String(world.age[i]));
        setText('leafs-eaten', String(world.harvested[i]));
        setText('leafs-kids', String(world.offspring[i]));
    }

    /* A tap on a forager follows it; a tap on empty ground adds one there,
     * with full fuel and a random heading, as the Viewer's click does. */
    function onTap(ev) {
        const rect = canvas.getBoundingClientRect();
        const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
        const x = (cx - ox) / scale, y = EXT_Y - (cy - oy) / scale;
        if (x < 0 || x >= EXT_X || y < 0 || y >= EXT_Y) return;
        const tol = Math.max(SELECT_R, SELECT_PX / scale);
        let best = -1, bestD2 = tol * tol;
        for (let j = 0; j < world.n; j++) {
            const dx = minImage(world.px[j] - x, EXT_X);
            const dy = minImage(world.py[j] - y, EXT_Y);
            const d2 = dx * dx + dy * dy;
            if (d2 <= bestD2) { bestD2 = d2; best = j; }
        }
        trail = [];
        if (best >= 0) {
            selected = world.ids[best];
        } else {
            world.add(x, y);
            selected = null;
        }
        if (!running) { paint(); readout(); }
    }

    /* ── Loop ───────────────────────────────────────────────────────────── */
    function frame() {
        if (running) {
            for (let s = 0; s < STEPS; s++) world.step();
            record();
            const i = selected === null ? -1 : world.indexOf(selected);
            if (i >= 0) {
                trail.push([world.px[i], world.py[i]]);
                if (trail.length > TRAIL) trail.shift();
            }
        }
        // Painted even when paused, so a window resize cannot blank the card.
        paint();
        drawTrace();
        drawPhase();
        readout();
        requestAnimationFrame(frame);
    }

    /* ── UI ─────────────────────────────────────────────────────────────── */
    const UI = `
      <div id="leafs-head">
        <span class="leafs-title">pyLEAFS</span>
        <button class="leafs-hbtn" id="leafs-reset">Reset</button>
        <button class="leafs-hbtn" id="leafs-pause">Pause</button>
      </div>
      <div id="leafs-stage">
        <canvas id="leafs-canvas"></canvas>
        <div id="leafs-readout" class="leafs-ov">
          <div class="leafs-row"><span class="k">time</span><span class="v" id="leafs-t">0.00</span></div>
          <div class="leafs-row"><span class="k">foragers</span><span class="v" id="leafs-na">5</span></div>
          <div class="leafs-row"><span class="k">resources</span><span class="v" id="leafs-nr">0</span></div>
          <div id="leafs-agent">
            <div class="leafs-row"><span class="k">forager</span><span class="v" id="leafs-id"></span></div>
            <div class="leafs-row"><span class="k">fuel</span><span class="v" id="leafs-fuel"></span></div>
            <div class="leafs-row"><span class="k">age</span><span class="v" id="leafs-age"></span></div>
            <div class="leafs-row"><span class="k">harvested</span><span class="v" id="leafs-eaten"></span></div>
            <div class="leafs-row"><span class="k">offspring</span><span class="v" id="leafs-kids"></span></div>
          </div>
        </div>
        <div id="leafs-controls" class="leafs-ov">
          <div class="leafs-ctl">
            <label for="leafs-xi"><span>homogeneity \\(\\Xi\\)</span> <span class="val" id="leafs-xi-val">0.50</span></label>
            <input type="range" id="leafs-xi" min="0.1" max="1.5" step="0.05" value="0.5">
          </div>
          <div class="leafs-ctl">
            <label for="leafs-sense"><span>sensing radius \\(R_\\mathrm{sense}\\)</span> <span class="val" id="leafs-sense-val">6.0</span></label>
            <input type="range" id="leafs-sense" min="1" max="10" step="0.5" value="6">
          </div>
          <div class="leafs-ctl">
            <label for="leafs-v"><span>speed \\(v\\)</span> <span class="val" id="leafs-v-val">20</span></label>
            <input type="range" id="leafs-v" min="2" max="40" step="1" value="20">
          </div>
          <div class="leafs-ctl">
            <label for="leafs-sigma"><span>heading noise \\(\\sigma\\)</span> <span class="val" id="leafs-sigma-val">0.10</span></label>
            <input type="range" id="leafs-sigma" min="0" max="2" step="0.05" value="0.1">
          </div>
        </div>
        <div id="leafs-meters" class="leafs-ov">
          <div class="leafs-meter">
            <canvas id="leafs-trace"></canvas>
            <div class="leafs-cap"><span class="ag">foragers</span> and <span class="rs">resources</span></div>
          </div>
          <div class="leafs-meter">
            <canvas id="leafs-phase"></canvas>
            <div class="leafs-cap">phase plane, log axes</div>
          </div>
        </div>
      </div>
    `;

    /* The docs load MathJax (sphinx.ext.mathjax), but it has already swept
     * the page by the time we inject this markup, so typeset by hand. */
    function typesetMath(el) {
        const M = window.MathJax;
        if (!M) return;
        if (M.startup && M.startup.promise) {
            M.startup.promise.then(() => M.typesetPromise([el])).catch(() => {});
        } else if (M.typesetPromise) {
            M.typesetPromise([el]).catch(() => {});
        }
    }

    /* A slider: read it into P, show it, and hand it to the world. */
    function slider(id, key, digits, apply) {
        document.getElementById(id).addEventListener('input', function () {
            P[key] = parseFloat(this.value);
            document.getElementById(id + '-val').textContent = P[key].toFixed(digits);
            apply(P[key]);
        });
    }

    function boot() {
        const root = document.getElementById('leafs-app');
        if (!root) return;
        root.innerHTML = UI;
        typesetMath(root);

        canvas = document.getElementById('leafs-canvas');
        traceCv = document.getElementById('leafs-trace');
        phaseCv = document.getElementById('leafs-phase');
        layout();
        if (window.ResizeObserver) new ResizeObserver(layout).observe(canvas);
        else window.addEventListener('resize', layout);
        canvas.addEventListener('click', onTap);

        newWorld();

        slider('leafs-xi', 'Xi', 2, v => world.setXi(v));
        slider('leafs-sense', 'R_sense', 1, v => { world.Rsense = v; });
        slider('leafs-v', 'v', 0, v => { world.v = v; });
        slider('leafs-sigma', 'sigma', 2, v => { world.sigma = v; });
        document.getElementById('leafs-reset').addEventListener('click', newWorld);
        document.getElementById('leafs-pause').addEventListener('click', function () {
            running = !running;
            this.textContent = running ? 'Pause' : 'Resume';
        });

        requestAnimationFrame(frame);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
