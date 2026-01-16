-- strip-yaml-frontmatter.lua
-- Removes YAML frontmatter artifacts from included files
-- This handles the case where {{< include >}} brings in files with YAML headers

-- Pattern to detect YAML-like content that leaked from included files
local yaml_patterns = {
    "^editor:",
    "^bibliography:",
    "^csl:",
    "^format:",
    "^---$"
}

-- Check if a string looks like leaked YAML frontmatter
local function is_yaml_artifact(text)
    for _, pattern in ipairs(yaml_patterns) do
        if text:match(pattern) then
            return true
        end
    end
    -- Also catch the joined YAML line pattern
    if text:match("^editor:.*render%-on%-save") then
        return true
    end
    return false
end

-- Filter for Para elements (where YAML content often ends up)
function Para(el)
    local text = pandoc.utils.stringify(el)
    if is_yaml_artifact(text) then
        return {} -- Remove the element
    end
    return el
end

-- Filter for Plain elements
function Plain(el)
    local text = pandoc.utils.stringify(el)
    if is_yaml_artifact(text) then
        return {}
    end
    return el
end

-- Filter for HorizontalRule (the --- delimiter)
-- We need to be careful here - only remove if it's likely a YAML delimiter
-- This is handled by removing it when it follows YAML-like content

-- Alternative approach: Filter at the block level to catch sequences
function Blocks(blocks)
    local result = {}
    local skip_next_hr = false

    for i, block in ipairs(blocks) do
        local dominated_by_yaml = false

        if block.t == "Para" or block.t == "Plain" then
            local text = pandoc.utils.stringify(block)
            if is_yaml_artifact(text) then
                dominated_by_yaml = true
                skip_next_hr = true
            end
        end

        -- Skip HorizontalRule if it follows YAML content (the closing ---)
        if block.t == "HorizontalRule" and skip_next_hr then
            skip_next_hr = false
            dominated_by_yaml = true
        end

        if not dominated_by_yaml then
            table.insert(result, block)
            skip_next_hr = false
        end
    end

    return result
end

-- Also handle Header elements that might be empty due to YAML stripping
function Header(el)
    local text = pandoc.utils.stringify(el)
    -- Remove empty section headers that result from YAML artifacts
    if text == "" or text:match("^%s*$") then
        return {}
    end
    return el
end
