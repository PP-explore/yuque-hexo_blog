const { matterMarkdownAdapter } = require('@elog/cli');
const yaml = require('js-yaml'); // 新增 YAML 解析库

/**
 * 自定义文档处理器
 * @param {DocDetail} doc doc 的类型定义为 DocDetail
 */
const format = (doc) => {
  // 1. 提取并解析 YAML 代码块
  const yamlBlockRegex = /^```yaml\n([\s\S]*?)\n```/;
  const match = doc.body.match(yamlBlockRegex);
  
  if (match) {
    try {
      // 解析 YAML 代码块 [2](@ref)
      const customFrontmatter = yaml.load(match[1]);
      
      // 2. 合并元数据（优先级：YAML 代码块 > 默认属性）
      doc.properties = {
        ...doc.properties,          // Elog 生成的默认属性
        ...customFrontmatter,        // 语雀文档中的自定义属性
        categories: mergeCategories(doc.properties, customFrontmatter) // 特殊处理分类
      };
      
      // 3. 从正文中移除 YAML 代码块
      doc.body = doc.body.replace(match[0], '').trim();
    } catch (e) {
      console.error('YAML 解析错误:', e.message);
    }
  }

  // 4. 转换语雀特殊标记（保留原有功能）
  if (doc.body) {
    const regexTips = /:::(?<type>tips+)\n(?<content>.+)\n:::/gi;
    doc.body = doc.body.replace(regexTips, (match, type, content) => {
      return `{% note default %}\n${content}\n{% endnote %}`;
    });

    const regexNote = /:::(?<type>[a-z]+)\n(?<content>.+)\n:::/gi;
    doc.body = doc.body.replace(regexNote, (match, type, content) => {
      return `{% note ${type} %}\n${content}\n{% endnote %}`;
    });
  }
  
  return matterMarkdownAdapter(doc);
};

// 特殊处理分类合并逻辑 [11](@ref)
function mergeCategories(defaultProps, customProps) {
  const defaultCats = Array.isArray(defaultProps.categories) 
    ? defaultProps.categories 
    : [defaultProps.categories].filter(Boolean);
  
  const customCats = Array.isArray(customProps.categories) 
    ? customProps.categories 
    : [customProps.categories].filter(Boolean);
  
  return [...new Set([...defaultCats, ...customCats])];
}

module.exports = {
  format,
};