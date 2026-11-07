// Shared style of the proof papers: Bourbaki-style numbered statements and Lamport-style hierarchical proofs.
// Use it with: #import "../style.typ": * and #show: paper.with(title: "...")

#let paper(title: none, body) = {
  set document(title: title, author: "EBaMl contributors")
  set page(paper: "a4", margin: 2.4cm, numbering: "1")
  set text(size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): set block(above: 1.6em, below: 0.9em)
  show raw: set text(size: 9pt)
  show link: set text(fill: rgb("#1a4f8b"))
  show table: set par(justify: false)
  show table.cell: set block(breakable: false)
  // Inline code in a table cell has no space to break at, so a long name can run into the next column.
  // Set it as monospace text with a zero-width space after each ".", "_", "(", "," and "/", which allows a line break.
  // The result is text and not raw, so the rule does not apply to its own output.
  show table: it => {
    show raw.where(block: false): code => text(
      font: ("DejaVu Sans Mono", "Liberation Mono"),
      size: 9pt,
      code.text.replace(regex("[._(,/]"), piece => piece.text + sym.zws),
    )
    it
  }
  body
}

// ---- Bourbaki-style numbered statements -----------------------------------------------------------------------

#let stmt(kind, name, body, lean: none, italic: true) = {
  counter(kind).step()
  block(width: 100%, above: 1em, below: 0.8em, breakable: false, {
    strong[#kind #context counter(kind).display().]
    if name != none [ (#name).]
    [ ]
    if italic { emph(body) } else { body }
    if lean != none {
      v(0.2em)
      set par(justify: false)
      text(size: 9pt, fill: rgb("#444444"))[Lean: #lean.slice(1).map(raw).join(", "), in #raw(lean.at(0)).]
    }
  })
}
#let notation(name, body) = stmt("Notation", name, body, italic: false)
#let definition(name, body, lean: none) = stmt("Definition", name, body, lean: lean, italic: false)
#let lemma(name, body, lean: none) = stmt("Lemma", name, body, lean: lean)
#let theorem(name, body, lean: none) = stmt("Theorem", name, body, lean: lean)
#let corollary(name, body, lean: none) = stmt("Corollary", name, body, lean: lean)

// ---- Lamport-style hierarchical proofs ------------------------------------------------------------------------

#let step(level, number, body) = block(
  width: 100%,
  above: 0.55em,
  below: 0.55em,
  inset: (left: (level - 1) * 1.6em),
  [$chevron.l #level chevron.r #number$ #h(0.5em) #body],
)
#let proof(body) = block(width: 100%, inset: (left: 1em), above: 0.3em, below: 1em, body)
#let qed(uses) = [*Q.E.D.* by #uses.]
