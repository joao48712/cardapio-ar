const STEPS: { title: string; items: string[] }[] = [
  {
    title: "1. Material",
    items: [
      "Qualquer celular recente (iPhone ou Android).",
      "App RealityScan (grátis) ou Polycam. Os dois rodam em iPhone e Android.",
      "Uma base giratória (prato de bolo giratório serve) e uma toalha fosca, cinza ou bege.",
    ],
  },
  {
    title: "2. Luz",
    items: [
      "Luz suave e igual dos lados: perto de uma janela sem sol direto, ou duas luminárias com papel vegetal na frente.",
      "Sem flash. Sem sombra dura. Sem reflexo forte no prato ou no molho.",
      "Pratos muito brilhantes ou transparentes (gelatina, vidro, calda) saem piores. Tente um pouco de luz mais difusa.",
    ],
  },
  {
    title: "3. Montagem",
    items: [
      "Monte o prato como sai para a mesa. Use o prato e a louça reais: eles entram no 3D.",
      "Coloque sobre a base giratória, no centro da toalha.",
      "Faça a captura logo depois de montar, antes de murchar ou derreter.",
    ],
  },
  {
    title: "4. Captura",
    items: [
      "Gire a base devagar e fotografe em 3 alturas: quase na linha da mesa, a 45° e de cima.",
      "60 a 100 fotos no total. Cada foto deve cobrir boa parte da anterior.",
      "Mantenha o prato inteiro no quadro e a câmera parada a cada clique.",
    ],
  },
  {
    title: "5. Exportar",
    items: [
      "No app, processe em qualidade média ou alta.",
      "Corte a toalha e a base giratória, deixando só o prato.",
      "Exporte em GLB. Se o app pedir, escolha textura de 2048 px.",
    ],
  },
  {
    title: "6. Subir no painel",
    items: [
      "Em Pratos, abra o prato e envie o .glb. O painel reduz o arquivo e põe a base no chão.",
      "Confira a largura em cm. Se não bater com o prato real, digite a medida e toque em Ajustar escala.",
      "Gire a prévia para o melhor ângulo: ela vira a imagem de capa. Depois salve.",
      "Teste no celular: abra o prato e toque em Ver na sua mesa.",
    ],
  },
];

export const instant = false;

export default function CaptureGuide() {
  return (
    <div className="max-w-2xl">
      <h1 className="font-display text-3xl">Guia de captura 3D</h1>
      <p className="mt-2 text-muted">
        Como fotografar um prato para virar 3D. Comece com 1 ou 2 pratos e teste no celular antes de fazer o cardápio todo.
      </p>
      <div className="mt-8 flex flex-col gap-8">
        {STEPS.map((s) => (
          <section key={s.title}>
            <h2 className="font-display text-2xl">{s.title}</h2>
            <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[15px] leading-relaxed text-muted">
              {s.items.map((i) => <li key={i}>{i}</li>)}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
