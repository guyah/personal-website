import { getCollection } from "astro:content";

const escapeXml = (value: string) =>
	value
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&apos;");

export async function GET(context: { site: URL }) {
	const siteUrl = context.site?.toString() ?? "https://example.com";
	const posts = (await getCollection("blog"))
		.filter((post) => !post.data.draft)
		.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

	const items = posts
		.map((post) => {
			const link = `${siteUrl.replace(/\/$/, "")}/blog/${post.slug}/`;
			return `\n  <item>\n    <title>${escapeXml(post.data.title)}</title>\n    <link>${link}</link>\n    <guid>${link}</guid>\n    <pubDate>${post.data.pubDate.toUTCString()}</pubDate>\n    <description>${escapeXml(post.data.description)}</description>\n  </item>`;
		})
		.join("");

	const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0">\n<channel>\n  <title>Guy Abi Hanna — Blog</title>\n  <description>Writing and notes on AI engineering, agents, and building.</description>\n  <link>${siteUrl}</link>${items}\n</channel>\n</rss>\n`;

	return new Response(xml, {
		headers: {
			"Content-Type": "application/rss+xml; charset=utf-8",
		},
	});
}
