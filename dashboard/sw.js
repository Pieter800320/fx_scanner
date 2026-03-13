const CACHE_NAME = 'fx-dashboard-v1';
const urlsToCache = ['./index.html', './manifest.json'];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  if (event.request.url.includes('_dashboard.json')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        return cached || fetch(event.request).then(network => {
          return caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, network.clone());
            return network;
          });
        });
      })
    );
  }
});
