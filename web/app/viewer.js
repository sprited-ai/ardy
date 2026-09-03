// three.js viewport in the demo's style: light background, floor grid, skinned Core mesh, skeleton, start arrow.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewer {
  constructor(container, skeleton, skin) {
    this.skeleton = skeleton; this.skin = skin; this.J = skeleton.parents.length; this.follow = true; this.dark = false;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }); this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement); this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200); this.camera.position.set(3.5, 2.2, 4.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement); this.controls.target.set(0, 0.9, 0); this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.0)); const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(4, 8, 3); this.scene.add(sun);
    this.grid = new THREE.GridHelper(40, 40); this.scene.add(this.grid);
    this.jointMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.03, 10, 8), new THREE.MeshStandardMaterial({ color: 0x3b6fd6 }), this.J); this.scene.add(this.jointMesh);
    this.bonePairs = []; skeleton.parents.forEach((p, i) => { if (p >= 0) this.bonePairs.push([i, p]); });
    const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.bonePairs.length * 6), 3));
    this.bones = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x224488 })); this.scene.add(this.bones);
    this.arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.02, 0), 0.6, 0x4444ff, 0.15, 0.1); this.scene.add(this.arrow);
    if (skin) {
      const geom = new THREE.BufferGeometry(); this.skinPos = new Float32Array(skin.V * 3); this.skinPos.set(skin.verts);
      geom.setAttribute('position', new THREE.BufferAttribute(this.skinPos, 3));
      geom.setIndex(new THREE.BufferAttribute(skin.faces instanceof Uint16Array ? skin.faces : new Uint32Array(skin.faces), 1)); geom.computeVertexNormals();
      this.skinMat = new THREE.MeshStandardMaterial({ color: 0xa9b8f0, roughness: 0.55, metalness: 0.05, transparent: true, opacity: 1.0 });
      this.skinMesh = new THREE.Mesh(geom, this.skinMat); this.scene.add(this.skinMesh); this.A = new Float32Array(this.J * 12);
    }
    this.setDark(false); this.showMesh = true; this.showSkeleton = false; this.m4 = new THREE.Matrix4(); this.v3 = new THREE.Vector3();
    addEventListener('resize', () => this.resize()); this.resize();
  }
  setDark(dark) { this.dark = dark; this.scene.background = new THREE.Color(dark ? 0x14171c : 0xffffff); this.scene.fog = new THREE.Fog(this.scene.background, 14, 40); this.grid.material.color.set(dark ? 0x3a4250 : 0xd0d4dc); this.grid.material.opacity = 0.9; this.grid.material.transparent = true; }
  resize() { const w = this.container.clientWidth, h = this.container.clientHeight; this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  setPose(frame) {
    if (!frame) return; const { joints, rots } = frame, J = this.J;
    this.jointMesh.visible = this.bones.visible = this.showSkeleton;
    if (this.showSkeleton) {
      const pos = this.bones.geometry.attributes.position.array;
      for (let j = 0; j < J; j++) { this.m4.makeTranslation(joints[j * 3], joints[j * 3 + 1], joints[j * 3 + 2]); this.jointMesh.setMatrixAt(j, this.m4); }
      this.jointMesh.instanceMatrix.needsUpdate = true;
      this.bonePairs.forEach(([a, b], k) => { pos[k * 6] = joints[a * 3]; pos[k * 6 + 1] = joints[a * 3 + 1]; pos[k * 6 + 2] = joints[a * 3 + 2]; pos[k * 6 + 3] = joints[b * 3]; pos[k * 6 + 4] = joints[b * 3 + 1]; pos[k * 6 + 5] = joints[b * 3 + 2]; });
      this.bones.geometry.attributes.position.needsUpdate = true;
    }
    if (this.skinMesh) { this.skinMesh.visible = this.showMesh; if (this.showMesh && rots) this.skinFrame(joints, rots); }
    this.lastRoot = [joints[0], joints[1], joints[2]];
    if (this.follow) { this.v3.set(joints[0], 0.9, joints[2]); const delta = this.v3.clone().sub(this.controls.target).multiplyScalar(0.08); this.controls.target.add(delta); this.camera.position.add(delta); }
  }
  setStartArrow(pos, headingRad) { this.arrow.position.set(pos[0], 0.02, pos[2]); this.arrow.setDirection(new THREE.Vector3(Math.cos(headingRad), 0, -Math.sin(headingRad))); }
  skinFrame(joints, rots) {   // linear blend skinning, same math as ardy/viz/core_skin.py
    const { V, W, inv, verts, idx, w } = this.skin, A = this.A, J = this.J, out = this.skinPos;
    for (let j = 0; j < J; j++) { const r = j * 9, t = j * 3, o = j * 16, a = j * 12;
      for (let row = 0; row < 3; row++) { const R0 = rots[r + row * 3], R1 = rots[r + row * 3 + 1], R2 = rots[r + row * 3 + 2], P = joints[t + row];
        for (let col = 0; col < 4; col++) A[a + row * 4 + col] = R0 * inv[o + col] + R1 * inv[o + 4 + col] + R2 * inv[o + 8 + col] + P * inv[o + 12 + col]; } }
    for (let v = 0; v < V; v++) { const x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2]; let ox = 0, oy = 0, oz = 0;
      for (let k = 0; k < W; k++) { const wk = w[v * W + k]; if (wk === 0) continue; const a = idx[v * W + k] * 12;
        ox += wk * (A[a] * x + A[a + 1] * y + A[a + 2] * z + A[a + 3]); oy += wk * (A[a + 4] * x + A[a + 5] * y + A[a + 6] * z + A[a + 7]); oz += wk * (A[a + 8] * x + A[a + 9] * y + A[a + 10] * z + A[a + 11]); }
      out[v * 3] = ox; out[v * 3 + 1] = oy; out[v * 3 + 2] = oz; }
    this.skinMesh.geometry.attributes.position.needsUpdate = true; this.skinMesh.geometry.computeVertexNormals();
  }
  render() { this.controls.update(); this.renderer.render(this.scene, this.camera); }
}
