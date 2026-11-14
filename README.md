# Cardápio AR

Cardápio do restaurante com realidade aumentada: o cliente vê o prato na mesa, no tamanho real, pela câmera do celular. Sem instalar app.
Next.js 16 + Supabase + `<model-viewer>`, hospedado na Vercel.

- Site: https://cardapio-ar-six.vercel.app
- Painel: https://cardapio-ar-six.vercel.app/admin
- Publicação: cada `git push` na `main` publica sozinho na Vercel.

## Rodar local

```bash
npm install
npm run dev
```

Sem `.env.local`, o site usa pratos de exemplo (`src/lib/seed.ts`) e o painel mostra os passos de configuração.

## Ligar o Supabase

1. Crie um projeto em supabase.com.
2. SQL Editor: rode `supabase/migrations/0001_init.sql`.
3. Authentication > Users: crie seu usuário (e-mail + senha).
4. SQL Editor: `insert into public.admins (email) values ('seu@email');`
5. Copie `.env.example` para `.env.local` e preencha URL e chave anon (Project Settings > API).
6. Reinicie `npm run dev` e entre em `/admin`.

Na Vercel, cadastre as mesmas variáveis em Settings > Environment Variables.

## Onde mexer

| O quê | Onde |
|---|---|
| Nome, descrição, logo | `src/lib/restaurant.ts` + arquivo em `public/brand/` |
| Cores | `src/app/globals.css` (bloco `:root`) |
| Fontes | `src/app/layout.tsx` |
| Pratos, preços, fotos, 3D | painel `/admin` |
| QR codes das mesas | `/admin/qr` |
| Como fotografar em 3D | `/admin/guia-captura` |

## Como o 3D funciona

- Upload de `.glb` no painel → otimização no navegador (`src/lib/optimize-glb.ts`): simplifica para ~60k triângulos, texturas JPEG até 2048 px, base do prato na origem, escala opcional em cm.
- Página do prato carrega o 3D só quando o cliente toca em "Ver em 3D".
- AR: Android abre no Scene Viewer; iPhone no Quick Look (USDZ gerado na hora pelo model-viewer).

## Modelo de exemplo

`public/seed/sobremesa.glb` é gerado por `node scripts/make-sample-glb.mjs`.
