// scripts/fix-sitemap.js
const fs = require('fs');
const path = './public/sitemap.xml';
if (!fs.existsSync(path)) {
  console.error('sitemap not found:', path);
  process.exit(1);
}
let xml = fs.readFileSync(path, 'utf8');

// 对 & < > " ' 进行 xml 转义，但避免重复转义已有的实体（如 &amp;）
xml = xml.replace(/&(?!(?:amp;|lt;|gt;|quot;|apos;))/g, '&amp;')
         .replace(/</g, '&lt;') // 视情况可选择是否对所有 < 转义（< 在正常xml标签里不应被转义）
         .replace(/>/g, '&gt;');

// 注意：通常只需要处理 & 和 < > 出现在文本或 loc 内时的问题。
// 如果你的 sitemap 中 < 和 > 只用于标签，请保守处理，避免破坏标签结构。

fs.writeFileSync(path, xml, 'utf8');
console.log('sitemap fixed.');
