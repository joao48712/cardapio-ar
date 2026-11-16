// PROTÓTIPO (descartável). Pratos com nomes vistos no Instagram do 153;
// foto e 3D emprestados do único prato real cadastrado até aqui.

export type ProtoDish = {
  slug: string;
  name: string;
  description: string;
  price: number;
  category: string;
  tags: string[];
  featured: boolean;
  pairings: string[];
  photo: string | null;
  model: string | null;
};

export type ProtoCategory = { slug: string; name: string };

export const CATEGORIES: ProtoCategory[] = [
  { slug: "entradas", name: "Entradas" },
  { slug: "principais", name: "Principais" },
  { slug: "sanduiches", name: "Sanduíches" },
  { slug: "sobremesas", name: "Sobremesas" },
  { slug: "drinks", name: "Drinks" },
];

export const BASE: Omit<ProtoDish, "photo" | "model">[] = [
  { slug: "trio-de-brusquetas", name: "Trio de brusquetas", description: "Pão de fermentação natural, tomate confitado, burrata e pesto de manjericão.", price: 4200, category: "entradas", tags: ["vegetariano"], featured: true, pairings: ["gin-tonica-da-casa"] },
  { slug: "carpaccio", name: "Carpaccio", description: "Lâminas de filé, alcaparras, parmesão e rúcula com azeite.", price: 4800, category: "entradas", tags: ["sem glúten"], featured: false, pairings: [] },
  { slug: "bife-ancho", name: "Bife ancho com fritas", description: "Ancho grelhado na brasa, batatas rústicas e o molho da casa.", price: 8900, category: "principais", tags: [], featured: true, pairings: ["petit-gateau-no-potinho"] },
  { slug: "risoto-de-cogumelos", name: "Risoto de cogumelos", description: "Arbóreo, mix de cogumelos frescos e manteiga trufada.", price: 7400, category: "principais", tags: ["vegetariano", "sem glúten"], featured: false, pairings: [] },
  { slug: "hamburger-153", name: "Hambúrguer 153", description: "Blend da casa 180 g, cheddar, cebola roxa e picles no pão brioche.", price: 4600, category: "sanduiches", tags: [], featured: true, pairings: ["gin-tonica-da-casa"] },
  { slug: "sanduiche-de-file", name: "Sanduíche de filé mignon", description: "Filé em tiras, tomate seco, rúcula e maionese de ervas.", price: 5200, category: "sanduiches", tags: ["picante"], featured: false, pairings: [] },
  { slug: "petit-gateau-no-potinho", name: "Petit gâteau no potinho", description: "Chocolate quente, brigadeiro cremoso e frutas vermelhas.", price: 3200, category: "sobremesas", tags: [], featured: true, pairings: [] },
  { slug: "gin-tonica-da-casa", name: "Gin tônica da casa", description: "Gin, tônica, laranja-bahia e alecrim queimado. Para o pôr do sol.", price: 3600, category: "drinks", tags: ["vegano"], featured: false, pairings: [] },
];


export const ORDER_URL = "https://restaurante153.saipos.com";
