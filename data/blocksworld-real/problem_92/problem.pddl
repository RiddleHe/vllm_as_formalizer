(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        yellow - block
        purple - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on green blue)
        (on yellow green)
        (on purple yellow)
        (on red purple)
        (on orange red)
        (clear orange)
        ; blue -> green -> yellow -> purple -> red -> orange
    )
    (:goal (and (on green purple)
        (on red green)
        (on blue yellow)
        (on orange blue)))
)
    