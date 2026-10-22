import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { createContext, SourceTextModule } from 'node:vm';

test('actual worker entry hashes packed geometry and computes Three normals without DOM, importmap or GPU', async () => {
  const bytes = new ArrayBuffer(48);
  new Float32Array(bytes, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  new Uint32Array(bytes, 36, 3).set([0, 1, 2]);
  const digest = Buffer.from(await webcrypto.subtle.digest('SHA-256', bytes)).toString('hex');
  const posted = [];
  const context = createContext({ URL, performance, crypto: webcrypto, Float32Array, Uint32Array, Uint8Array,
    fetch: async url => {
      assert.ok(url.startsWith('/api/mb-sonic/assets/'));
      return { ok: true, arrayBuffer: async () => bytes.slice(0) };
    }, self: { postMessage: (value, transfers) => posted.push({ value, transfers }) } });
  const paths = new Map([
    ['/protocol.mjs', new URL('./protocol.mjs', import.meta.url)],
    ['/vendor/three.module.js', new URL('../full_corpus_lab/vendor/three.module.js', import.meta.url)],
  ]);
  const modules = new Map();
  async function load(path, url) {
    if (!modules.has(path)) {
      const module = new SourceTextModule(await readFile(url, 'utf8'), { context,
        initializeImportMeta: meta => { meta.url = `http://127.0.0.1:28948${path}`; },
        importModuleDynamically: () => assert.fail('Worker must not import browser OrbitControls') });
      modules.set(path, module);
      await module.link(async specifier => {
        const resolved = new URL(specifier, `http://127.0.0.1:28948${path}`).pathname;
        assert.ok(paths.has(resolved), `Unexpected worker dependency: ${specifier}`);
        return load(resolved, paths.get(resolved));
      });
    }
    return modules.get(path);
  }
  const module = await load('/app.js?mesh-worker', new URL('./app.js', import.meta.url));
  await module.evaluate();
  await context.self.onmessage({ data: { asset_id: digest, vertex_count: 3, index_count: 3 } });
  assert.equal(posted[0].value.error, undefined);
  assert.deepEqual([...posted[0].value.normals], [0, 0, 1, 0, 0, 1, 0, 0, 1]);
  assert.equal(posted[0].transfers.length, 2);
  assert.ok(Math.abs(posted[0].value.sphere.radius - Math.SQRT1_2) < 1e-8);
  await context.self.onmessage({ data: { asset_id: '0'.repeat(64), vertex_count: 3, index_count: 3 } });
  assert.match(posted[1].value.error, /content hash mismatch/);
  assert.equal(posted[1].transfers, undefined);
});
