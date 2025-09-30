(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        purple - block
        green - block
        blue - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on purple yellow)
        (on green purple)
        (on blue green)
        (on red blue)
        (on orange red)
        (clear orange)
        ; yellow -> purple -> green -> blue -> red -> orange
    )
    (:goal (and (on green orange)
        (on red blue)
        (on yellow red)
        (on purple yellow)))
)
    